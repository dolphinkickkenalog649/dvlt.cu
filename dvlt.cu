// dvlt.cu
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "include/dvlt_preprocess.h"
#include "include/arena.h"
#include "include/pos_interp.h"
#include "include/ply_writer.h"
#include "kernels/dvlt_model.cuh"
#include "include/cuda_util.h"

static bool is_directory(const char* p) { struct stat s; return stat(p, &s) == 0 && S_ISDIR(s.st_mode); }

// print the hand-made logo (assets/logo.txt) in cuda-green; terminal only, silently
// skipped if the file isn't found. the saved report.txt uses a plain "dvlt.cu" title.
static void print_banner() {
    FILE* f = fopen("assets/logo.txt", "r");
    if (!f) return;
    fputs("\033[32m", stdout);
    char buf[256];
    while (fgets(buf, sizeof buf, f)) fputs(buf, stdout);
    fclose(f);
    fputs("github.com/yassa9\033[0m\n", stdout);
}

// collect jpg/jpeg/png from a directory, sorted (so frame order is deterministic).
static std::vector<std::string> expand_dir(const char* dir) {
    std::vector<std::string> out;
    DIR* d = opendir(dir);
    if (!d) { fprintf(stderr, "cannot open directory %s\n", dir); return out; }
    for (dirent* e; (e = readdir(d)); ) {
        std::string n = e->d_name, l = n; std::transform(l.begin(), l.end(), l.begin(), ::tolower);
        auto ends = [&](const char* s){ size_t k = strlen(s); return l.size() >= k && l.compare(l.size()-k, k, s) == 0; };
        if (ends(".jpg") || ends(".jpeg") || ends(".png"))
            out.push_back(std::string(dir) + "/" + n);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

static void usage(const char* prog) {
    fprintf(stderr,
        "usage: %s [options] <directory | img1 img2 [img3 ...]>\n\n"
        "options:\n"
        "  -o, --output <dir>     output directory (default: output/<timestamp>)\n"
        "  -w, --weights <path>   weights file     (default: model/weights.dvlt)\n"
        "  -c, --conf <0-1>       drop the lowest-confidence fraction of points (default: 0)\n"
        "  -s, --img-size <N>     working resolution (default: 504)\n"
        "  -i, --input <img>      add an input image (repeatable; input is also positional)\n"
        "  -b, --no-banner        hide the ascii logo banner\n"
        "  -h, --help             show this help\n\n"
        "outputs:\n"
        "  <output>/scene.ply     merged point cloud (all views, world space)\n"
        "  <output>/poses.json    camera poses (extrinsics c2w + intrinsics per view)\n\n"
        "examples:\n"
        "  %s photos/                      all jpg/png in directory\n"
        "  %s img1.jpg img2.jpg            explicit image list\n"
        "  %s -o results/ -c 0.5 photos/   custom output + confidence drop\n\n"
        "video: extract frames first, then run on the directory:\n"
        "  ./tools/v2f.sh -i clip.mp4 -s   sample sharp frames -> inputs/clip/\n",
        prog, prog, prog, prog);
}

int main(int argc, char** argv) {
    // weights default to the converter's output path; override with -w.
    std::string weights_path = "model/weights.dvlt";
    const char* output_dir = nullptr;   // default below: output/<timestamp>
    int img_size = 504;                 // released working resolution (config.json img_size)
    float conf_frac = 0.f;              // drop this bottom fraction of points by confidence
    bool show_banner = true;            // -b/--no-banner suppresses the terminal logo
    std::vector<std::string> positional;
    auto opt = [&](int i, const char* s, const char* l){ return !strcmp(argv[i], s) || !strcmp(argv[i], l); };
    for (int i = 1; i < argc; i++) {
        if (opt(i, "-h", "--help")) { usage(argv[0]); return 0; }
        else if (opt(i, "-b", "--no-banner")) show_banner = false;
        else if (opt(i, "-o", "--output")   && i + 1 < argc) output_dir   = argv[++i];
        else if (opt(i, "-w", "--weights")  && i + 1 < argc) weights_path = argv[++i];
        else if (opt(i, "-c", "--conf")     && i + 1 < argc) conf_frac    = atof(argv[++i]);
        else if (opt(i, "-s", "--img-size") && i + 1 < argc) img_size     = atoi(argv[++i]);
        else if (opt(i, "-i", "--input")    && i + 1 < argc) positional.push_back(argv[++i]);
        else if (argv[i][0] == '-') { fprintf(stderr, "unknown option: %s\n", argv[i]); usage(argv[0]); return 1; }
        else positional.push_back(argv[i]);
    }
    // default output dir: output/YYYY_MM_DD_HH_MM_SS.
    char ts_buf[64];
    if (!output_dir) {
        time_t now = time(nullptr); struct tm* lt = localtime(&now);
        snprintf(ts_buf, sizeof ts_buf, "output/%04d_%02d_%02d_%02d_%02d_%02d",
                 lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday, lt->tm_hour, lt->tm_min, lt->tm_sec);
        output_dir = ts_buf;
    }
    // a single directory arg expands to its images; otherwise the positionals are the image list.
    std::vector<std::string> imgs;
    if (positional.size() == 1 && is_directory(positional[0].c_str())) imgs = expand_dir(positional[0].c_str());
    else imgs = positional;
    if (imgs.empty()) { usage(argv[0]); return 1; }

    auto t_wall = std::chrono::high_resolution_clock::now();
    auto ms_since = [](auto t0) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    };
    // print to stdout AND tee into `report` (ansi escapes stripped) -> written to report.txt later.
    // the report drops the leading 2-space shift on each line (kept on the terminal) so the file
    // is flush-left; deeper sub-item indent is preserved (only the base shift is removed).
    std::string report;
    auto emit = [&](const char* s){
        fputs(s, stdout); fflush(stdout);
        int indent = (report.empty() || report.back() == '\n') ? 2 : 0;
        for (const char* p = s; *p; ) {
            if (*p == '\033' && p[1] == '[') {                       // skip a CSI escape (e.g. color)
                p += 2; while (*p && !((*p>='a'&&*p<='z')||(*p>='A'&&*p<='Z'))) p++; if (*p) p++;
                continue;
            }
            char c = *p++;
            if (c == '\n') { report += c; indent = 2; continue; }
            if (indent > 0 && c == ' ') { indent--; continue; }      // strip the base shift
            indent = 0;
            report += c;
        }
    };
    auto line = [&](const char* fmt, auto... a){ char b[128]; snprintf(b, sizeof b, fmt, a...); emit(b); };
    auto row  = [&](const char* k, double ms){ line("  %-22s %10.1f\n", k, ms); };
    auto rows = [&](const char* k, const char* v){ line("  %-22s %10s\n", k, v); };
    const char* rule = "──────────────────────";

    constexpr int B = 1, D = 768, H = 12, DH = 64, R = 4, PATCH_START = 5;
    constexpr int ENC_BLOCKS = 12, DEC_DIM = 384, DEC_HEADS = 6, DEC_DEPTH = 2;
    constexpr int LOOP_STEPS = 12, PATCH = 14, M = 37;   // M = native dinov2 grid the blob was dumped at
    dvlt::DvltConfig cfg{};
    cfg.B = B; cfg.D = D; cfg.H = H; cfg.DH = DH; cfg.R = R; cfg.patch_start = PATCH_START;
    cfg.enc_blocks = ENC_BLOCKS; cfg.dec_dim = DEC_DIM; cfg.dec_heads = DEC_HEADS; cfg.dec_depth = DEC_DEPTH;
    cfg.loop_steps = LOOP_STEPS; cfg.patch_size = PATCH;
    dvlt::DvltConfig cfg_native = cfg;
    cfg_native.Himg = M * PATCH; cfg_native.Wimg = M * PATCH; cfg_native.ph = M; cfg_native.pw = M;
    cfg_native.Sp = M * M; cfg_native.P = PATCH_START + M * M;

    // overlap the cpu preprocess (worker thread, no cuda) with the weight load (main thread):
    // independent since the weights parse at the native 37x37 grid. each records its own duration.
    double prep_ms = 0, weight_ms = 0;
    dvltpp::Batch batch;
    auto t_pp = std::chrono::high_resolution_clock::now();
    std::thread pp([&]{ batch = dvltpp::preprocess(imgs, img_size, 14); prep_ms = ms_since(t_pp); });

    auto t_wt = std::chrono::high_resolution_clock::now();
    CublasCtx cublas; cublas.init();
    dvlt::WeightFile wf; wf.open(weights_path.c_str());
    dvlt::DvltWeights w{};
    dvlt::load_dvlt_weights(w, wf, cfg_native);   // correct cursor walk at native Sp
    weight_ms = ms_since(t_wt);
    pp.join();

    const int S = batch.S, Himg = batch.H, Wimg = batch.W;
    const int ph = Himg / 14, pw = Wimg / 14, Sp = ph * pw;
    // interpolate the encoder pos embed from the native MxM grid to the cropped ph x pw grid.
    if (ph != M || pw != M) w.enc.pos = dvlt::interpolate_pos_embed(w.pos_native, M, D, ph, pw);

    // streamed report: title, an info block (config/value), then a timing block (stage/time).
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); char v[64];
    if (show_banner) print_banner();                    // green logo, terminal only
    report += "\ndvlt.cu\ngithub.com/yassa9\n";          // plain title in the saved report
    line("\n  %-22s %10s\n", "info", "value");
    rows(rule, "──────────");
    snprintf(v, sizeof v, "sm_%d", prop.major*10 + prop.minor);     rows("arch", v);
    snprintf(v, sizeof v, "%zu MB", prop.totalGlobalMem >> 20);     rows("vram total", v);
    line("  %-22s %10d\n", "frames", S);
    snprintf(v, sizeof v, "%dx%d", Wimg, Himg);                     rows("resolution", v);
    snprintf(v, sizeof v, "%dx%d", pw, ph);                         rows("patch grid", v);
    snprintf(v, sizeof v, "%.2f", conf_frac);                       rows("conf drop", v);
    rows(rule, "──────────");
    line("  %-22s %10s\n", "stage", "time (ms)");
    rows(rule, "──────────");
    row("weights loading", weight_ms);
    row("preprocess", prep_ms);   // runs concurrently with the weight load (overlapped in wall time)

    // pipeline init (sizes the arena to free vram).
    auto t = std::chrono::high_resolution_clock::now();
    cfg.S = S; cfg.Himg = Himg; cfg.Wimg = Wimg; cfg.ph = ph; cfg.pw = pw; cfg.Sp = Sp; cfg.P = cfg.patch_start + Sp;
    dvlt::DvltPipeline pipe;
    pipe.init(cublas, cfg, /*arena_bytes=*/0, /*max_len=*/std::max(ph, pw) + 4);
    double init_ms = ms_since(t); row("pipeline init", init_ms);
    float* img_dev = dvt::to_dev_f32(batch.norm.data(), (size_t)S * 3 * Himg * Wimg);

    // gpu stages: sync after each so the row streams (and the time is real).
    double gpu_total = 0;
    auto stage = [&](const char* name, auto&& fn) {
        auto ts = std::chrono::high_resolution_clock::now();
        fn(); CUDA_CHECK(cudaDeviceSynchronize());
        double dt = ms_since(ts); gpu_total += dt; row(name, dt);
    };
    stage("encoder (12 blk)",   [&]{ pipe.run_encoder(w, img_dev); });
    stage("loop (12 steps)",    [&]{ pipe.run_loop(w); });
    stage("ray + camera head",  [&]{ pipe.run_ray_cam(w); });
    stage("depth head",         [&]{ pipe.run_depth(w); });
    stage("world points",       [&]{ pipe.run_finalize(); });
    stage("ransac pose",        [&]{ pipe.solve_pose(); });

    // vram now at its peak (arena + weights + buffers all resident, nothing freed yet).
    size_t fb, tb; cudaMemGetInfo(&fb, &tb);
    size_t vram_peak = (tb - fb) >> 20, vram_total = tb >> 20;

    // read back, filter, write outputs (one timed "output" row).
    auto t_out = std::chrono::high_resolution_clock::now();
    const size_t hw = (size_t)Himg * Wimg;
    auto world = dvt::from_dev_f32(pipe.world, (size_t)S * hw * 3);
    auto depth = dvt::from_dev_f32(pipe.depth, (size_t)S * hw);
    auto conf  = dvt::from_dev_f32(pipe.conf,  (size_t)S * hw);
    auto extr  = dvt::from_dev_f32(pipe.extrinsics(), (size_t)S * 16);
    auto intr  = dvt::from_dev_f32(pipe.intrinsics(), (size_t)S * 9);

    // keep finite, positive-depth points; with --conf P>0 also drop the lowest-P fraction by
    // confidence (percentile threshold over the finite/positive points).
    float conf_thresh = -1e30f;
    if (conf_frac > 0.f) {
        std::vector<float> cv;
        for (size_t i = 0; i < (size_t)S * hw; i++)
            if (std::isfinite(conf[i]) && depth[i] > 0.f && depth[i] < 1e4f) cv.push_back(conf[i]);
        if (!cv.empty()) {
            size_t k = (size_t)(conf_frac * cv.size()); if (k >= cv.size()) k = cv.size() - 1;
            std::nth_element(cv.begin(), cv.begin() + k, cv.end());
            conf_thresh = cv[k];
        }
    }
    std::vector<uint8_t> mask((size_t)S * hw, 0);
    size_t kept = 0;
    for (size_t i = 0; i < (size_t)S * hw; i++) {
        float d = depth[i];
        bool ok = std::isfinite(world[i*3]) && std::isfinite(world[i*3+1]) && std::isfinite(world[i*3+2])
                  && d > 0.f && d < 1e4f && conf[i] >= conf_thresh;
        mask[i] = ok ? 1 : 0; kept += ok;
    }
    mkdir("output", 0755);
    mkdir(output_dir, 0755);
    std::string ply_stem = std::string(output_dir) + "/scene";
    write_pointcloud_ply_filtered(ply_stem.c_str(), world.data(), mask.data(), batch.norm.data(), S, Himg, Wimg);

    // poses.json: per-frame extrinsics (c2w 4x4) + intrinsics (3x3).
    std::string jpath = std::string(output_dir) + "/poses.json";
    FILE* jf = fopen(jpath.c_str(), "w");
    fprintf(jf, "{\n  \"frames\": [\n");
    for (int s = 0; s < S; s++) {
        const float* E = extr.data() + (size_t)s * 16;
        const float* K = intr.data() + (size_t)s * 9;
        fprintf(jf, "    {\n      \"extrinsics_c2w\": [");
        for (int k = 0; k < 16; k++) fprintf(jf, "%s%.6f", k ? ", " : "", E[k]);
        fprintf(jf, "],\n      \"intrinsics\": [");
        for (int k = 0; k < 9; k++) fprintf(jf, "%s%.6f", k ? ", " : "", K[k]);
        fprintf(jf, "]\n    }%s\n", s + 1 < S ? "," : "");
    }
    fprintf(jf, "  ]\n}\n");
    fclose(jf);
    double post_ms = ms_since(t_out);

    // footer: aggregate timing + vram + summary
    // compute = sum of the gpu stage rows; wall clock = whole run (disk load -> ply written).
    rows(rule, "──────────");
    row("compute",     gpu_total);
    row("postprocess", post_ms);
    row("wall clock",  ms_since(t_wall));
    rows(rule, "──────────");
    snprintf(v, sizeof v, "%zu MB", vram_peak);  rows("vram peak", v);
    snprintf(v, sizeof v, "%zu MB", vram_total); rows("vram total", v);

    const size_t total_pts = (size_t)S * hw;
    line("\n  %s/\n", output_dir);
    line("    %-16s %zu pts  (%d%% kept)\n", "scene.ply", kept, (int)(100.0 * kept / total_pts));
    line("    %-16s %d poses\n", "poses.json", S);
    line("    %-16s the report above\n\n", "report.txt");

    // dump the full report (as printed, minus colors) next to the ply/json.
    std::string rpath = std::string(output_dir) + "/report.txt";
    if (FILE* rf = fopen(rpath.c_str(), "w")) { fputs(report.c_str(), rf); fclose(rf); }
    return 0;
}
