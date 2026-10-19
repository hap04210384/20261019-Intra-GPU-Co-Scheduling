#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <atlstr.h>
#include <string>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <ppl.h>
#include <windows.h>  
#include <chrono>
#include <sstream>
#include <climits>
#include <atlconv.h> 
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include <atomic>
#include <mma.h>

//=================================================================================================================
//CString transSetFile = _T("..\\TransactionSets\\TCGA_BRCA_transactions.txt"); double supportThreshold = 0.77;


//CString transSetFile = _T("..\\TransactionSets\\chess.txt"); double supportThreshold = 0.95;
//CString transSetFile = _T("..\\TransactionSets\\connect.txt"); double supportThreshold = 0.965;
//CString transSetFile = _T("..\\TransactionSets\\mushroom.txt"); double supportThreshold = 0.52;
//CString transSetFile = _T("..\\TransactionSets\\T10I4D100K.txt"); double supportThreshold = 0.0225;
//CString transSetFile = _T("..\\TransactionSets\\retail.txt"); double supportThreshold = 0.00125;

CString transSetFile = _T("..\\TransactionSets\\pumsb_x256.txt"); double supportThreshold = 0.8;
//CString transSetFile = _T("..\\TransactionSets\\pumsb_star.txt"); double supportThreshold = 0.275;
//CString transSetFile = _T("..\\TransactionSets\\pumsb.txt"); double supportThreshold = 0.8;
//CString transSetFile = _T("..\\TransactionSets\\T40I10D100K.txt"); double supportThreshold = 0.025;
//CString transSetFile = _T("..\\TransactionSets\\kosarak.txt"); double supportThreshold = 0.0024;

double frequencyThreshold;//double is necessary!
int RUN_MODE = 1;// 消融开关: 0=纯串行CPU, 1=仅GPU, 2=CPU+GPU异构协同(默认)
//=================================================================================================================
struct FI {
    std::vector<int> itemsSet;
    int frequency;
    int root;
    int progress;
};

std::vector<std::vector<int>> h_transSet;
int* d_transSet;

int dimension;
int transNum;
int itemIDmax, itemIDmin;
size_t transLengthMax;
size_t transLengthMin;
double transLengthMean;

int transNumReduced;
int dimensionReduced;
size_t transLengthReducedMin;
size_t transLengthReducedMax;
double transLengthReducedMean;

std::vector<int> freqPerItem;
std::vector<int> freqPerItem_inSort;
std::vector<int> items_inFreqSort;

std::vector<FI> FIsStack;
std::mutex mtxFIsStackk;

std::vector<FI> MFIsPool;
std::mutex mtxMFIsPool;

double preprocessing_time;
double computation_time;

//=================================================================================================================
int CPUphysicalCores;
int CPUlogicalCores;
double GPUtaskPercentage;

__device__ cudaError_t cudaStatus;
__device__ int* d_TransSetReducedRAM;
//=================================================================================================================
// 根据计算能力获取每个SM的CUDA核心数
static int getCoresPerSM(int major, int minor) {
    switch (major)
    {
    case 2: return (minor == 0) ? 32 : 48; // Fermi
    case 3: return 192; // Kepler
    case 5: return 128; // Maxwell
    case 6: return 128; // Pascal
    case 7: return 64;  // Volta
    case 8: return 128;  // Ampere
    case 9: return 128; // Ada Lovelace
    case 10: return 192; // Blackwell (示例，需以NVIDIA官方数据为准)
    default: return -1; // 未知架构
    }
}

CString getVSversion() {
    CString vsInfo;
    if (_MSC_VER >= 1940) vsInfo = _T("2022 (v143)");  // VS2022 17.10+ 仍归属v143
    else if (_MSC_VER >= 1930) vsInfo = _T("2022 (v143)");  // VS2022 17.0~17.9
    else if (_MSC_VER >= 1920) vsInfo = _T("2019 (v142)");  // VS2019 16.x
    else if (_MSC_VER >= 1910) vsInfo = _T("2017 (v141)");  // VS2017 15.x
    else if (_MSC_VER == 1900) vsInfo = _T("2015 (v140)");  // VS2015 14.0
    else if (_MSC_VER == 1800) vsInfo = _T("2013 (v120)");  // VS2013 12.0
    else if (_MSC_VER == 1700) vsInfo = _T("2012 (v110)");  // VS2012 11.0
    else if (_MSC_VER == 1600) vsInfo = _T("2010 (v100)");  // VS2010 10.0
    else {
        CString strMSCVer;
        strMSCVer.Format(_T("%d"), _MSC_VER);
        vsInfo = _T("Unknown Visual Studio Version (MSC_VER = ") + strMSCVer + _T(")");
    }
    return vsInfo;
}
//=================================================================================================================
void getCPUInfo(CString& cpuName, int& physicalCores, int& logicalCores) {
    // 1. 初始化参数
    cpuName = _T("Unknown CPU");  // 宽字符兼容，适配Unicode
    physicalCores = 0;
    logicalCores = 0;

    // 2. 获取逻辑核心数
    SYSTEM_INFO sysInfo{};
    GetSystemInfo(&sysInfo);
    logicalCores = static_cast<int>(sysInfo.dwNumberOfProcessors);

    // 3. 获取物理核心数（简化容错）
    DWORD bufferSize = 0;
    GetLogicalProcessorInformation(NULL, &bufferSize);
    if (bufferSize > 0) {
        char* buffer = new (std::nothrow) char[bufferSize];
        if (buffer && GetLogicalProcessorInformation((SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer, &bufferSize)) {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION* info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION*)buffer;
            DWORD infoCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
            for (DWORD i = 0; i < infoCount; i++) {
                if (info[i].Relationship == RelationProcessorCore) {
                    physicalCores++;
                }
            }
        }
        delete[] buffer;
    }
    // 容错：获取失败则物理核心数=逻辑核心数
    physicalCores = (physicalCores == 0) ? logicalCores : physicalCores;

    // 4. 获取CPU名称（转CString）
    int cpuInfo[4] = { 0 };
    char cpuBrand[0x40] = { 0 };  // 临时存储ASCII格式CPU名称
    __cpuid(cpuInfo, 0x80000000);
    if (cpuInfo[0] >= 0x80000004) {
        // 分3次读取CPU名称
        __cpuid(cpuInfo, 0x80000002); memcpy(cpuBrand + 0, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000003); memcpy(cpuBrand + 16, cpuInfo, 16);
        __cpuid(cpuInfo, 0x80000004); memcpy(cpuBrand + 32, cpuInfo, 16);

        // 转换为CString（自动处理ASCII→Unicode）
        cpuName = CString(cpuBrand);
        // 修剪前后多余空格（美化）
        cpuName.Trim();
    }
}

static void displayWorkingEnvironment() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount); // Get and display GPU information
    std::cout << "======================================================================" << std::endl;
    for (int i = 0; i < deviceCount; ++i)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        std::cout << "GPU " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Multi-Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Number of Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
        int coresPerSM = getCoresPerSM(deviceProp.major, deviceProp.minor);
        if (coresPerSM == -1) {
            std::cout << "Unknown architecture, unable to calculate the number of CUDA cores !" << std::endl;
        }
        else {
            int totalCores = coresPerSM * deviceProp.multiProcessorCount;
            std::cout << "  Number of CUDA Cores Per SM: " << coresPerSM << std::endl;
            std::cout << "  Total Number of CUDA Cores: " << totalCores << std::endl;
        }
        std::cout << std::endl;
    }

    CString cpuName;
    getCPUInfo(cpuName, CPUphysicalCores, CPUlogicalCores);
    std::cout << "CPU: " << cpuName << std::endl;
    {
        std::cout << "  Number of Physical Cores: " << CPUphysicalCores << std::endl;
        std::cout << "  Number of Logical Cores: " << CPUlogicalCores << std::endl << std::endl;
    }

    std::cout << "Visual Studio: " << getVSversion() << ", MSC_VER: " << _MSC_VER << std::endl;
    int rt_ver = 0;
    cudaError_t err = cudaRuntimeGetVersion(&rt_ver);
    if (err == cudaSuccess) std::cout << "CUDA Version: " << rt_ver / 1000 << "." << (rt_ver % 1000) / 10 << std::endl;
    else std::cout << "CUDA Version not detected: " << cudaGetErrorString(err) << std::endl;
}
//=================================================================================================================
static bool readTransSetFile(CString transSetFile) {
    FILE* fp = _tfopen(transSetFile, _T("rb"));
    if (!fp) { std::cout << transSetFile << " 文件打开失败 ！" << std::endl; return false; }
    _fseeki64(fp, 0, SEEK_END); long long sz = _ftelli64(fp); _fseeki64(fp, 0, SEEK_SET);
    std::vector<char> buf(sz);
    if (sz > 0 && fread(buf.data(), 1, sz, fp) != (size_t)sz) { fclose(fp); return false; }
    fclose(fp);
    // fast parse: whitespace-separated ints, one transaction per line, negatives skipped
    // (semantics identical to the original getline + istringstream version)
    h_transSet.reserve((size_t)(sz / 40) + 16);
    const char* p = buf.data(); const char* end = p + sz;
    std::vector<int> row; row.reserve(128);
    while (p < end) {
        char c = *p;
        if (c == '\n') { h_transSet.push_back(row); row.clear(); p++; continue; }
        if (c >= '0' && c <= '9') {
            long v = 0;
            while (p < end && *p >= '0' && *p <= '9') v = v * 10 + (*p++ - '0');
            row.push_back((int)v);
            continue;
        }
        if (c == '-') {  // negative item id: consume digits, drop item (as before)
            p++;
            while (p < end && *p >= '0' && *p <= '9') p++;
            continue;
        }
        p++;
    }
    if (!row.empty()) h_transSet.push_back(row);

    itemIDmin = INT_MAX;
    itemIDmax = 0;
    transNum = 0;
    transLengthMin = INT_MAX;
    transLengthMax = 0;
    transLengthMean = 0;

    long long len = 0;
    for (auto& row : h_transSet) {
        for (int x : row) {
            if (x > itemIDmax) itemIDmax = x;
            if (x < itemIDmin) itemIDmin = x;
        }
        len = row.size();
        if (len < transLengthMin) transLengthMin = len;
        if (len > transLengthMax) transLengthMax = len;
        transLengthMean += len;
        ++transNum;
    }
    if (transNum > 0) transLengthMean /= transNum;

    dimension = 0;
    std::vector<int> itemList(itemIDmax + 1, 0);//to get dimension
    for (auto& row : h_transSet) {
        for (int x : row) {
            if (itemList[x] == 0) {
                dimension++;
                itemList[x]++;
            }
        }
    }
    return true;
}

static void getItemsFrequenc() {
    freqPerItem.insert(freqPerItem.end(), itemIDmax + 1, 0);
    for (auto& row : h_transSet) {
        for (int x : row) {
            freqPerItem[x]++;
        }
    }
    freqPerItem_inSort = freqPerItem;
    items_inFreqSort.resize(freqPerItem.size());
    std::iota(items_inFreqSort.begin(), items_inFreqSort.end(), 0);
    // 稳定排序替代冒泡：O(d log d)，并保持等频项的原始次序（与冒泡一致）
    std::stable_sort(items_inFreqSort.begin(), items_inFreqSort.end(),
        [&](int a, int b) { return freqPerItem[a] > freqPerItem[b]; });
    for (size_t i = 0; i < items_inFreqSort.size(); i++)
        freqPerItem_inSort[i] = freqPerItem[items_inFreqSort[i]];

    /*
    std::cout << "======================================================================" << std::endl;
    std::cout << "freqPerItem: ";
    for (int val : freqPerItem) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << std::endl << "freqPerItem_inSort: ";
    for (int val : freqPerItem_inSort) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << std::endl << "items_inFreqSort: ";
    for (int val : items_inFreqSort) std::cout << val << " ";
    std::cout << std::endl;*/
}

static bool FIsStackkPop(FI* result) { // 安全出栈操作
    std::lock_guard<std::mutex> lock(mtxFIsStackk);
    if (!FIsStack.empty())
    {
        *result = FIsStack.back();
        FIsStack.pop_back();
        return true;
    }
    else return false;
}
static void FIsStackkPush(FI* pushOne) { // 安全入栈操作
    std::lock_guard<std::mutex> lock(mtxFIsStackk);
    FIsStack.push_back(*pushOne);
}

static void MFIsPoolPush(FI* pushOne) { // 安全入栈操作
    std::lock_guard<std::mutex> lock(mtxMFIsPool);
    MFIsPool.push_back(*pushOne);
}

static bool isOutside(int item, FI* one)
{
    int i;
    for (i = 0; i < one->itemsSet.size(); i++) {
        if (item == one->itemsSet[i]) break;
    }
    if (i < one->itemsSet.size()) return false;
    else return true;
}

//===================== 位图（binary vector）支持度统计：全局结构 =====================
// 每个项一个位向量：第 t 位置 1 表示第 t 条（精减后）事务包含该项。
// 候选 X 的条件库 = X 中各项位向量的 AND；项 e 在其中的频度 = popcount(AND 结果 & bv[e])。
// 与原先逐事务扫描完全等价，但把 O(N×L) 的比较变成 O(N/64) 的位运算。
static std::vector<unsigned long long> h_bitmap;   // (itemIDmax+1) × bitmapWords 的扁平数组
static std::vector<int> denseToOrig;   // ID compaction: dense id -> original item id (for output unmapping)
static int bitmapWords = 0;
static std::vector<int> freqItems;                 // 精减后幸存项（freqPerItem >= frequencyThreshold）的 ID 列表
static int freqItemsNum = 0;
static unsigned long long* d_bitmap = nullptr;     // GPU 端位图（只读，所有工作线程共享）
static int* d_freqItems = nullptr;                 // GPU 端幸存项 ID 列表

// BMMA (Tensor Core b1) lookahead batch counting: globals (impl. below, before recursiveExpansion)
static unsigned* d_bmmaB = nullptr;        // compact B: freqItems (padded) x W32 uint32 words
static int bmmaW32 = 0;                    // = bitmapWords * 2 (uint32 words per N-bit vector)
static int bmmaDpad = 0;                   // freqItemsNum rounded up to a multiple of 64
static const int BMMA_BATCH_MIN = 64;      // below this many children -> per-node fallback
static const size_t BMMA_ABYTES_CAP = 96ull << 20;  // per-wave A budget: ~100MB is the measured optimum across shapes (ablation 2026-09-25, Table VI)
// --- ablation overrides (2026-09-25): env-driven, no recompile needed ---
// BMMA_WAVE_CAP=N  : force wave size to N rows (64-aligned, clamped to [64,1024]); unset/0 = default byte-budget rule
// BMMA_SPLITK_OFF=1: disable split-k (single-slice k loop) for the on/off ablation
static int envWaveCap() {
    const char* e = getenv("BMMA_WAVE_CAP");
    int v = (e && *e) ? atoi(e) : 0;
    return v > 0 ? std::min(1024, std::max(64, v / 64 * 64)) : 0;
}
static bool envSplitkOff() {
    const char* e = getenv("BMMA_SPLITK_OFF");
    return e && *e && atoi(e) != 0;
}
// BMMA_MASK_WPB=N : override mask-build words-per-block quota (default 4096 = MASK_WPB)
static int envMaskWpb() {
    const char* e = getenv("BMMA_MASK_WPB");
    int v = (e && *e) ? atoi(e) : 0;
    return v >= 256 ? v : 4096;
}

// 运行时探测 SM 数：调度阈值按卡自适应，跨平台（sm_80+）无需改代码
static int gpuSMCount() {
    static int n = 0;
    if (n == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        if (n <= 0) n = 38;   // 查询失败时退回开发机（3060 Ti）的默认值
    }
    return n;
}
static std::atomic<long> g_bmmaBatches{ 0 };
static std::atomic<long> g_bmmaRows{ 0 };
static std::atomic<long> g_bmmaSmemBatches{ 0 };
static std::atomic<long> g_bmmaFallbackWaves{ 0 };
static double g_tMask = 0, g_tBmma = 0, g_tCopy = 0, g_tScatter = 0;   // wave engine is single-threaded
int BMMA_ENGINE = 1;   // RUN_MODE==1 && BMMA_ENGINE==1 -> wave engine (batched BMMA counting)

// ==================== L1 ROUTER: measured-cost-table co-scheduling ====================
// Mode D (ROUTE_MODE=1): per-wave routing between SIMT (CUDA cores) and BMMA (tensor cores),
//   decision from a measured cost table (EMA refreshed after every executed wave, Sec. 5.3/F).
// Mode E (ROUTE_MODE=2): same routing + two-stream pipeline: mask build of wave i+1 on the
//   SIMT stream overlaps the BMMA of wave i on the tensor stream (Thm. 3 steady state).
// Mode R (ROUTE_MODE=3): random seeded routing (decisions ignore costs) — exactness stress:
//   outputs must stay bit-identical under arbitrary routing history.
// ROUTE_ORACLE=1: execute BOTH engines for every wave and log both actual times (E12, small sets).
// Cold-start prior: E02-calibrated knee — waves with >=64 nodes go to the tensor engine
//   (knee measured at b~=32-64, d=256, N=1M); the EMA table takes over from the first
//   executed wave of each side.
static int envRouteMode() { const char* s = getenv("ROUTE_MODE"); return s ? atoi(s) : 0; }
static int envRouteSeed() { const char* s = getenv("ROUTE_SEED"); return s ? atoi(s) : 1; }
static double envKscale() { const char* s = getenv("ROUTE_KSCALE"); return (s && *s) ? atof(s) : 1.0; }
static double envHyst()   { const char* s = getenv("ROUTE_HYST");  return (s && *s) ? atof(s) : 0.10; }
static double envEngage() { const char* s = getenv("ROUTE_ENGAGE_MS"); return (s && *s) ? atof(s) : 0.0; }
static int envOracle()    { const char* s = getenv("ROUTE_ORACLE"); return s ? atoi(s) : 0; }
// cost model: "affine" (default) = per-wave fixed cost + per-node slope, online least
//   squares over all waves of one engine; "linear" = legacy per-node bucketed EMA.
static const char* envCostModel() { const char* s = getenv("ROUTE_COSTMODEL"); return (s && *s) ? s : "affine"; }
static bool useAffine() { return _stricmp(envCostModel(), "affine") == 0; }

// Affine cost model (Sec. 5.3): t(w) = a + b*w, fit online by least squares over every
// executed wave of one engine. The intercept a captures the per-wave fixed cost (kernel
// launch, mask build / thread-pool dispatch) that the legacy per-node model misses — it
// is exactly what makes small waves BMMA-dark (E12/E14). Slope is global; with fewer
// than MINFIT waves we fall back to the legacy bucketed table.
struct AffineCost {
    double sw = 0, st = 0, sww = 0, swt = 0; long n = 0;
    void add(int w, double t) { sw += w; st += t; sww += (double)w * w; swt += (double)w * t; n++; }
    bool fit(double& a, double& b) const {
        const long MINFIT = 2;   // two calibration probes per engine suffice (exact line)
        if (n < MINFIT) return false;
        double den = n * sww - sw * sw;
        if (den <= 0.0) return false;
        b = (n * swt - sw * st) / den;
        a = (st - b * sw) / n;
        if (a < 0.0) { a = 0.0; b = st / sw; }   // clamp to non-negative intercept
        return true;
    }
    double est(int w) const { double a, b; return fit(a, b) ? (a + b * w) : -1.0; }
};
static AffineCost g_acS, g_acB;
static const char* envRouteLog() { const char* s = getenv("ROUTE_LOG"); return (s && *s) ? s : "router_log.csv"; }

// Measured cost table (Sec. 5.3): per-node rates, bucketed by wave size so the
// tensor path's fixed per-wave cost is not extrapolated linearly. Each bucket
// holds an EMA of measured ms/node for each engine; the first executed wave of
// each engine is warmup (one-time cudaMalloc/module-init) and never enters the
// table.
static const int N_BUCKETS = 4;
static double g_rateSimt[N_BUCKETS] = { -1, -1, -1, -1 };   // ms/node, SIMT path
static double g_rateBmma[N_BUCKETS] = { -1, -1, -1, -1 };   // ms/node, tensor path
static double g_rateSimtAll = -1.0, g_rateBmmaAll = -1.0;   // fallback across buckets
static bool   g_warmSimt = false, g_warmBmma = false;
static double g_tDecide = 0;            // total decision overhead (ms)
static double g_tCollect = 0, g_tExpand = 0;    // wave collection / node expansion (host, ms)
static long   g_wavesSimt = 0, g_wavesTensor = 0, g_wavesTensorSmall = 0;
static std::ofstream g_routeLog;

static int waveBucket(int w) {
    if (w < 64) return 0;
    if (w < 256) return 1;
    if (w < 768) return 2;
    return 3;
}
static double estCost(int w, const double* buckets, double all) {
    int b = waveBucket(w);
    if (buckets[b] >= 0.0) return w * buckets[b];
    if (all >= 0.0) return w * all;
    return -1.0;
}
static void routeLogOpen() {
    g_routeLog.open(envRouteLog());
    g_routeLog << "wave_idx,w,kmax,freqItems,W32,est_simt_ms,est_bmma_ms,decision,actual_ms,decide_us,warmup";
    if (envOracle()) g_routeLog << ",oracle_simt_ms,oracle_bmma_ms";
    g_routeLog << "\n";
}
static void routeLogLine(long idx, int w, int kmax, double estS, double estB, int decision,
                         double actualMs, double decideUs, bool warmup, double oS = -1, double oB = -1) {
    if (!g_routeLog) return;
    g_routeLog << idx << "," << w << "," << kmax << "," << freqItemsNum << "," << bmmaW32
               << "," << estS << "," << estB << "," << (decision ? "tensor" : "simt")
               << "," << actualMs << "," << decideUs << "," << (warmup ? 1 : 0);
    if (envOracle()) g_routeLog << "," << oS << "," << oB;
    g_routeLog << "\n" << std::flush;
}

// per-wave routing decision. Returns 1 = tensor engine, 0 = SIMT engine.
// Hysteresis (Thm. 4): relative margin m = ROUTE_HYST; keep current side unless the
// other side wins by more than the margin.
static int routeDecision(int w, int kmax, int lastDecision) {
    auto t0 = std::chrono::high_resolution_clock::now();
    int d = 1;
    // Cold start (E13 finding): no forced SIMT probe. Probing a large first wave
    // on the CPU path costs 5-10x the tensor cost (synthetic sweep, Sec. 6.4);
    // instead route wave 0 by the E02 knee prior like any other wave — tensor for
    // w>=64 (its warmup absorbs one-time init), SIMT for w<64, which doubles as
    // the SIMT cost probe exactly where SIMT is cheap and actually wins.
    static int g_probeWaves = 0;
    if (envRouteMode() != 3 && g_probeWaves < 1) { g_probeWaves++; d = (w >= 128) ? 1 : 0; }
    else if (envRouteMode() == 3) {   // random seeded stress: decision independent of costs
        static unsigned long long rngState = (unsigned long long)envRouteSeed() * 6364136223846793005ULL + 1442695040888963407ULL;
        rngState = rngState * 6364136223846793005ULL + 1442695040888963407ULL;
        d = (int)((rngState >> 33) & 1ULL);
    } else {
        double estS = useAffine() ? g_acS.est(w) : -1.0;
        double estB = useAffine() ? g_acB.est(w) : -1.0;
        if (estS < 0.0) estS = estCost(w, g_rateSimt, g_rateSimtAll);   // legacy fallback
        if (estB < 0.0) estB = estCost(w, g_rateBmma, g_rateBmmaAll);
        if (estB >= 0.0) estB *= envKscale();
        if (estS >= 0.0 && estB >= 0.0) {
            d = (estB <= estS) ? 1 : 0;
            if (lastDecision >= 0 && d != lastDecision) {
                double m = envHyst();
                bool toTensor = (d == 1);
                bool justified = toTensor ? (estB < estS * (1.0 - m)) : (estS < estB * (1.0 - m));
                if (!justified) d = lastDecision;
            }
        } else if (estS >= 0.0) {
            if (envEngage() > 0.0) {
                // optional conservative engagement guard (default off; swept in E11):
                // wake the tensor engine only when the SIMT alternative is expensive
                d = (estS >= envEngage() && w >= 64) ? 1 : 0;
            } else {
                // no engagement guard: trust the measured table + knee prior.
                // Knee stays at the E02-calibrated 64 here: pumsb_x256's waves are
                // all capped at 64 nodes, so this branch is its only path to the
                // tensor engine (E13). Dark-zone safety at small w comes from the
                // 128 knee at cold start and from measured costs once both sides
                // have at least one wave.
                d = (w >= 64) ? 1 : 0;
            }
        } else {
            // SIMT cost unknown. Never gamble an unbounded CPU wave on a guess:
            // once the tensor side has a measured rate, trust it (E13: a blind
            // SIMT wave at N=4M cost 142 ms vs 3 ms on tensor cores). Only when
            // NEITHER side is measured (right after a tensor warm-up wave) fall
            // back to the knee prior; a small wave then doubles as the SIMT probe.
            if (estB >= 0.0) d = 1;
            else d = (w >= 128) ? 1 : 0;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    g_tDecide += std::chrono::duration<double>(t1 - t0).count() * 1000.0;
    return d;
}
// returns true if this measurement is warmup (excluded from the cost table)
static bool routeLearn(int decision, int w, double actualMs) {
    const double alpha = 0.4;   // EMA refresh per executed wave (Sec. 5.3/F: refreshed per round)
    if (decision == 0) {
        // SIMT per-node cost is genuine from the very first wave (only a tiny
        // stream/buffer init hides in it); learn immediately.
        double per = actualMs / std::max(1, w);
        int b = waveBucket(w);
        g_rateSimt[b] = (g_rateSimt[b] < 0.0) ? per : (1 - alpha) * g_rateSimt[b] + alpha * per;
        g_rateSimtAll = (g_rateSimtAll < 0.0) ? per : (1 - alpha) * g_rateSimtAll + alpha * per;
        g_acS.add(w, actualMs);
        return false;
    } else {
        if (!g_warmBmma) { g_warmBmma = true; return true; }   // first tensor wave pays cudaMalloc etc.
        double per = actualMs / std::max(1, w);
        int b = waveBucket(w);
        g_rateBmma[b] = (g_rateBmma[b] < 0.0) ? per : (1 - alpha) * g_rateBmma[b] + alpha * per;
        g_rateBmmaAll = (g_rateBmmaAll < 0.0) ? per : (1 - alpha) * g_rateBmmaAll + alpha * per;
        g_acB.add(w, actualMs);
    }
    return false;
}


static void buildBitmap() {
    bitmapWords = (int)((transNumReduced + 2047) / 2048 * 32);  // pad to 2048 bits: smem BMMA k-chunk = 2048 bits; padding bits are zero, counts unaffected
    // --- ID compaction (webdocs fix): surviving items are remapped to dense
    // ids 0..d-1 so the bitmap is sized by survivor count, not by the raw
    // id namespace (webdocs: 5.27M distinct ids would need ~1.1 TB).
    // Pure relabeling: mining semantics unchanged; outputs unmapped at print.
    freqItems.clear();
    for (int i = 1; i <= itemIDmax; i++) {
        if (freqPerItem[i] >= frequencyThreshold) freqItems.push_back(i);
    }
    freqItemsNum = (int)freqItems.size();
    denseToOrig = freqItems;
    std::vector<int> origToDense(itemIDmax + 1, -1);
    for (int k = 0; k < freqItemsNum; k++) origToDense[freqItems[k]] = k;
    const bool zeroFrequent = (freqPerItem.size() > 0 && freqPerItem[0] >= frequencyThreshold);
    if (zeroFrequent) { origToDense[0] = freqItemsNum; denseToOrig.push_back(0); }  // id 0 counts in masks but (as before) is never a candidate
    for (auto& row : h_transSet)
        for (int& x : row) x = origToDense[x];   // rows hold only frequent items after reduction
    const int dCount = freqItemsNum + (zeroFrequent ? 1 : 0);
    std::vector<int> freqDense(dCount + 1, 0);
    for (int k = 0; k < freqItemsNum; k++) freqDense[k] = freqPerItem[freqItems[k]];
    if (zeroFrequent) freqDense[freqItemsNum] = freqPerItem[0];
    freqPerItem.swap(freqDense);
    items_inFreqSort.resize(freqPerItem.size());
    std::iota(items_inFreqSort.begin(), items_inFreqSort.end(), 0);
    std::stable_sort(items_inFreqSort.begin(), items_inFreqSort.end(),
        [&](int a, int b) { return freqPerItem[a] > freqPerItem[b]; });  // ties: ascending dense id == ascending original id, same as before
    freqPerItem_inSort.assign(items_inFreqSort.size(), 0);
    for (size_t i = 0; i < items_inFreqSort.size(); i++)
        freqPerItem_inSort[i] = freqPerItem[items_inFreqSort[i]];
    for (int k = 0; k < freqItemsNum; k++) freqItems[k] = k;   // runtime (bitmap row indexing, GPU d_freqItems) uses dense ids; originals preserved in denseToOrig
    itemIDmax = dCount - 1;   // dense namespace: ids 0..dCount-1, sentinel slots preserved by +2 allocations
    h_bitmap.assign((size_t)(itemIDmax + 1) * bitmapWords, 0ULL);
    for (int t = 0; t < transNumReduced; t++) {
        for (int x : h_transSet[t]) {
            h_bitmap[(size_t)x * bitmapWords + (t >> 6)] |= (1ULL << (t & 63));
        }
    }
}

static void uploadBitmapToGPU() {
    size_t bitmapBytes = (size_t)(itemIDmax + 1) * bitmapWords * sizeof(unsigned long long);
    if (cudaMalloc(&d_bitmap, bitmapBytes) != cudaSuccess) { std::cerr << "d_bitmap分配失败" << std::endl; return; }
    if (cudaMemcpy(d_bitmap, h_bitmap.data(), bitmapBytes, cudaMemcpyHostToDevice) != cudaSuccess) { std::cerr << "d_bitmap拷贝失败" << std::endl; return; }
    if (cudaMalloc(&d_freqItems, freqItemsNum * sizeof(int)) != cudaSuccess) { std::cerr << "d_freqItems分配失败" << std::endl; return; }
    if (cudaMemcpy(d_freqItems, freqItems.data(), freqItemsNum * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) { std::cerr << "d_freqItems拷贝失败" << std::endl; return; }
    // BMMA compact B: survivor bitmaps only, col-major (one contiguous N-bit vector per item),
    // columns padded to a multiple of 64 (zero padding does not affect counts).
    bmmaW32 = bitmapWords * 2;
    bmmaDpad = (freqItemsNum + 63) / 64 * 64;
    std::vector<unsigned> hB((size_t)bmmaDpad * bmmaW32, 0u);
    const unsigned* srcAll = reinterpret_cast<const unsigned*>(h_bitmap.data());
    for (int c = 0; c < freqItemsNum; c++)
        memcpy(&hB[(size_t)c * bmmaW32], srcAll + (size_t)freqItems[c] * bmmaW32, (size_t)bmmaW32 * 4);
    if (cudaMalloc(&d_bmmaB, hB.size() * 4) != cudaSuccess) { std::cerr << "d_bmmaB alloc fail" << std::endl; return; }
    if (cudaMemcpy(d_bmmaB, hB.data(), hB.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess) { std::cerr << "d_bmmaB copy fail" << std::endl; return; }
}

static void getResFrequency(FI newOne, std::vector<int>* resFrequency) {
    std::fill(resFrequency->begin(), resFrequency->end(), 0);
    const int W = bitmapWords;
    static thread_local std::vector<unsigned long long> mask;  // 每线程复用，避免反复分配
    mask.assign(W, ~0ULL);
    for (int x : newOne.itemsSet) {
        const unsigned long long* bv = h_bitmap.data() + (size_t)x * W;
        for (int w = 0; w < W; w++) mask[w] &= bv[w];
    }
    for (int e : freqItems) {
        const unsigned long long* bv = h_bitmap.data() + (size_t)e * W;
        unsigned long long s = 0;
        for (int w = 0; w < W; w++) s += __popcnt64(mask[w] & bv[w]);
        resFrequency->at(e) = (int)s;
    }
    int cnt = 0;
    for (int e : freqItems) if (resFrequency->at(e) >= frequencyThreshold) cnt++;
    resFrequency->at(itemIDmax + 1) = cnt;
}

// 位图版核函数：每个 block 负责一个幸存项 e，块内线程按字跨步计算
// popcount(bv[e] & bv[X0] & bv[X1] & ...)，warp shuffle + shared memory 两级归约。
__global__ void getResFrequencyCudaKernel(int* d_resFrequencyCompact, int* d_itemsSet, int itemsNum,
    unsigned long long* d_bitmap, int bitmapWords, int* d_freqItems, int freqItemsNum)
{
    int eIdx = blockIdx.x;
    if (eIdx >= freqItemsNum) return;
    int e = d_freqItems[eIdx];
    unsigned long long* bvE = d_bitmap + (size_t)e * bitmapWords;

    unsigned long long local = 0;
    for (int w = threadIdx.x; w < bitmapWords; w += blockDim.x) {
        unsigned long long m = bvE[w];
        for (int i = 0; i < itemsNum; i++)
            m &= d_bitmap[(size_t)d_itemsSet[i] * bitmapWords + w];
        local += __popcll(m);
    }
    // warp 内归约
    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffffULL, local, offset);
    // warp 间归约
    __shared__ unsigned long long warpSums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = local;
    __syncthreads();
    if (warp == 0) {
        int numWarps = (blockDim.x + 31) >> 5;
        unsigned long long v = (lane < numWarps) ? warpSums[lane] : 0ULL;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffffULL, v, offset);
        if (lane == 0) d_resFrequencyCompact[eIdx] = (int)v;
    }
}

// 每个工作线程各自的常驻设备缓冲区：懒分配一次、复用全程，进程退出时由驱动统一回收。
// 消除了原先每次调用都 cudaMalloc/cudaFree 的巨额开销（实测每次调用约 1~2 ms，
// 而 chess 上核函数本身只需微秒级）。
static thread_local int* tl_d_resFrequency = nullptr;
static thread_local int* tl_d_itemsSet = nullptr;
static thread_local cudaStream_t tl_stream = nullptr;

cudaError_t getResFrequencyCuda(FI newOne, std::vector<int>* resFrequency) {
    // 前置校验：避免空指针/非法参数
    if (resFrequency == nullptr || resFrequency->empty() || newOne.itemsSet.empty()) {
        std::cerr << "错误：输入参数为空或无效" << std::endl;
        return cudaErrorInvalidValue;
    }
    if (itemIDmax <= 0 || bitmapWords <= 0 || freqItemsNum <= 0 || d_bitmap == nullptr || d_freqItems == nullptr) {
        std::cerr << "错误：位图未初始化或全局变量非法" << std::endl;
        return cudaErrorInvalidValue;
    }

    // 1. 初始化输出vector
    std::fill(resFrequency->begin(), resFrequency->end(), 0);

    // 2. 懒分配本线程常驻缓冲区、非阻塞流
    //    位图版结果采用紧凑布局：只有 freqItemsNum 个频度值，回传量从 (itemIDmax+1) 个 int 降到幸存项数个
    if (tl_d_resFrequency == nullptr) {
        cudaStatus = cudaMalloc(&tl_d_resFrequency, freqItemsNum * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "分配d_resFrequency失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_d_resFrequency = nullptr;
            return cudaStatus;
        }
    }
    if (tl_d_itemsSet == nullptr) {
        cudaStatus = cudaMalloc(&tl_d_itemsSet, (itemIDmax + 1) * sizeof(int));
        if (cudaStatus != cudaSuccess) {
            std::cerr << "分配d_itemsSet失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_d_itemsSet = nullptr;
            return cudaStatus;
        }
    }
    if (tl_stream == nullptr) {
        cudaStatus = cudaStreamCreateWithFlags(&tl_stream, cudaStreamNonBlocking);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "创建CUDA流失败：" << cudaGetErrorString(cudaStatus) << std::endl;
            tl_stream = nullptr;
            return cudaStatus;
        }
    }

    // 3. 仅拷贝候选项集本身（几十字节）
    int itemsNum = newOne.itemsSet.size();
    cudaStatus = cudaMemcpyAsync(tl_d_itemsSet, newOne.itemsSet.data(),
        itemsNum * sizeof(int), cudaMemcpyHostToDevice, tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "拷贝d_itemsSet失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 4. 每个 block 负责一个幸存项，256 线程/块，在本线程私有流上启动
    getResFrequencyCudaKernel << <freqItemsNum, 256, 0, tl_stream >> > (
        tl_d_resFrequency, tl_d_itemsSet, itemsNum, d_bitmap, bitmapWords, d_freqItems, freqItemsNum
        );
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "核函数启动失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 5. 紧凑结果在同一私有流上拷回，随后只等待本流完成
    static thread_local std::vector<int> h_compact;
    h_compact.resize(freqItemsNum);
    cudaStatus = cudaMemcpyAsync(h_compact.data(), tl_d_resFrequency,
        freqItemsNum * sizeof(int), cudaMemcpyDeviceToHost, tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "拷贝结果回主机端失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }
    cudaStatus = cudaStreamSynchronize(tl_stream);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "核函数执行失败：" << cudaGetErrorString(cudaStatus) << std::endl;
        return cudaStatus;
    }

    // 6. 散射回 resFrequency（按幸存项 ID），并统计达到阈值的项数
    int cnt = 0;
    for (int idx = 0; idx < freqItemsNum; idx++) {
        int v = h_compact[idx];
        resFrequency->at(freqItems[idx]) = v;
        if (v >= frequencyThreshold) cnt++;
    }
    resFrequency->at(itemIDmax + 1) = cnt;

    return cudaSuccess;
}

//===================== BMMA (Tensor Core b1) lookahead batch counting =====================
// Batched path (memo section 4 mapping): when node X expands into b >= BMMA_BATCH_MIN children,
// stack the child masks (mask(X) & bv[i]) into matrix A and evaluate ALL children's resFrequency
// rows with ONE b1 BMMA call against the compact survivor-bitmap matrix B (zero-copy layout match
// with h_bitmap). Children below the batch threshold fall back to the per-node path (memo risk #1).

using namespace nvcuda;

// A row = AND of the k listed item bitmaps (layout-identical reinterpretation of d_bitmap)
__global__ void maskBuildKernel(unsigned* __restrict__ A, const unsigned* __restrict__ Bmap,
    const int* __restrict__ items, int k, int W32)
{
    const int i = blockIdx.x;
    unsigned* a = A + (size_t)i * W32;
    const int* list = items + (size_t)i * k;
    for (int w = threadIdx.x; w < W32; w += blockDim.x) {
        unsigned x = ~0u;
        for (int t = 0; t < k; t++) x &= __ldg(Bmap + (size_t)list[t] * W32 + w);
        a[w] = x;
    }
}

// Faster mask build: 64-bit words (halves the instruction count; W32 is a multiple of 64
// so W64 = W32/2 divides exactly) + intra-row blocking (grid.y splits one row's word range
// across many blocks, raising occupancy when bcur alone cannot fill the SMs).
#define MASK_WPB 4096   // u64 words of one row handled per block
__global__ void maskBuildKernelFast(unsigned long long* __restrict__ A64,
    const unsigned long long* __restrict__ Bmap64,
    const int* __restrict__ items, int k, int W64, int wpb)
{
    const int i = blockIdx.x;
    const int w0 = blockIdx.y * wpb;
    const int w1 = min(W64, w0 + wpb);
    unsigned long long* a = A64 + (size_t)i * W64;
    const int* list = items + (size_t)i * k;
    for (int w = w0 + threadIdx.x; w < w1; w += blockDim.x) {
        unsigned long long x = ~0ull;
        for (int t = 0; t < k; t++) x &= __ldg(Bmap64 + (size_t)list[t] * W64 + w);
        a[w] = x;
    }
}

__global__ void bmmaKernelNaive(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
    int* __restrict__ C, int b, int d, int W32)
{
    const int warpId = threadIdx.x / 32;
    const int row0 = blockIdx.x * 64 + warpId * 8;
    const int col0 = blockIdx.y * 64;
    if (row0 >= b) return;
    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);
    const unsigned* aBase = A + (size_t)row0 * W32;
    for (int kw = 0; kw < W32; kw += 4) {
        wmma::load_matrix_sync(a_frag, aBase + kw, W32 * 32);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(b_frag, B + (size_t)(col0 + j * 8) * W32 + kw, W32 * 32);
            wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
        }
    }
    #pragma unroll
    for (int j = 0; j < 8; j++)
        if (col0 + j * 8 < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col0 + j * 8, c_frag[j], d, wmma::mem_row_major);
}

#define BMMA_TILE_K_WORDS 64
#define BMMA_TILE_K_BITS  (BMMA_TILE_K_WORDS * 32)
__global__ void bmmaKernelSmem(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
    int* __restrict__ C, int b, int d, int W32)
{
    __shared__ unsigned As[64][BMMA_TILE_K_WORDS];
    __shared__ unsigned Bs[64][BMMA_TILE_K_WORDS];
    const int warpId = threadIdx.x / 32;
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * 64 + warpId * 8;
    const int col0 = blockIdx.y * 64;
    const bool active = (row0 < b);   // inactive warps must still reach __syncthreads
    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);
    const unsigned* gA = A + (size_t)(blockIdx.x * 64) * W32;
    const unsigned* gB = B + (size_t)(blockIdx.y * 64) * W32;
    for (int kw = 0; kw < W32; kw += BMMA_TILE_K_WORDS) {
        #pragma unroll
        for (int idx = tid; idx < 64 * BMMA_TILE_K_WORDS; idx += 256) {
            const int r = idx / BMMA_TILE_K_WORDS, c = idx % BMMA_TILE_K_WORDS;
            As[r][c] = __ldg(gA + (size_t)r * W32 + kw + c);
            Bs[r][c] = __ldg(gB + (size_t)r * W32 + kw + c);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BMMA_TILE_K_WORDS; k += 4) {
            wmma::load_matrix_sync(a_frag, &As[warpId * 8][k], BMMA_TILE_K_BITS);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                wmma::load_matrix_sync(b_frag, &Bs[j * 8][k], BMMA_TILE_K_BITS);
                wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                    wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; j++)
        if (active && col0 + j * 8 < d)
            wmma::store_matrix_sync(C + (size_t)row0 * d + col0 + j * 8, c_frag[j], d, wmma::mem_row_major);
}

// Split-k variant for long-skinny geometry: with b=320/d=64 the (x,y) grid has only 5 blocks,
// far below the 38 SMs. grid.z partitions the k dimension; each split owns a private C slab
// (stride splitStride elements) and writes a full partial sum. A tiny reduction kernel then
// adds the slabs into slab 0. The k loop strides by kSplits tiles so any W32 works.
// (b1 accumulator fragments have an opaque element->(row,col) mapping, so cross-block
// accumulation must go through memory, not atomics on fragments.)
__global__ void bmmaKernelSmemSplitK(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
    int* __restrict__ C, int b, int d, int W32, int kSplits, long splitStride)
{
    __shared__ unsigned As[64][BMMA_TILE_K_WORDS];
    __shared__ unsigned Bs[64][BMMA_TILE_K_WORDS];
    const int warpId = threadIdx.x / 32;
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * 64 + warpId * 8;
    const int col0 = blockIdx.y * 64;
    const int z = blockIdx.z;
    const bool active = (row0 < b);
    wmma::fragment<wmma::matrix_a, 8, 8, 128, wmma::experimental::precision::b1, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 128, wmma::experimental::precision::b1, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 8, 8, 128, int> c_frag[8];
    #pragma unroll
    for (int j = 0; j < 8; j++) wmma::fill_fragment(c_frag[j], 0);
    const unsigned* gA = A + (size_t)(blockIdx.x * 64) * W32;
    const unsigned* gB = B + (size_t)(blockIdx.y * 64) * W32;
    for (int kw = z * BMMA_TILE_K_WORDS; kw < W32; kw += kSplits * BMMA_TILE_K_WORDS) {
        #pragma unroll
        for (int idx = tid; idx < 64 * BMMA_TILE_K_WORDS; idx += 256) {
            const int r = idx / BMMA_TILE_K_WORDS, c = idx % BMMA_TILE_K_WORDS;
            As[r][c] = __ldg(gA + (size_t)r * W32 + kw + c);
            Bs[r][c] = __ldg(gB + (size_t)r * W32 + kw + c);
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BMMA_TILE_K_WORDS; k += 4) {
            wmma::load_matrix_sync(a_frag, &As[warpId * 8][k], BMMA_TILE_K_BITS);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                wmma::load_matrix_sync(b_frag, &Bs[j * 8][k], BMMA_TILE_K_BITS);
                wmma::bmma_sync(c_frag[j], a_frag, b_frag, c_frag[j],
                    wmma::experimental::bmmaBitOpAND, wmma::experimental::bmmaAccumulateOpPOPC);
            }
        }
        __syncthreads();
    }
    int* Cout = C + (size_t)z * splitStride;
    #pragma unroll
    for (int j = 0; j < 8; j++)
        if (active && col0 + j * 8 < d)
            wmma::store_matrix_sync(Cout + (size_t)row0 * d + col0 + j * 8, c_frag[j], d, wmma::mem_row_major);
}

// C[i] = sum_z Csplits[z*splitStride + i] for i < n, reduced in place into slab 0.
__global__ void bmmaSplitReduceKernel(int* __restrict__ C, int kSplits, long splitStride, long n)
{
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int s = 0;
    for (int z = 0; z < kSplits; z++) s += C[(size_t)z * splitStride + i];
    C[i] = s;
}
// A row r = AND of node r's item bitmaps; C row r gives sup(node_r U {e}) for every survivor e.
// Item lists are padded to kmax by repeating items[0] (AND is idempotent).
// Returns per-node resFrequency vectors (size itemIDmax+2, last slot = freq-ext count).
static void batchCountBMMA(const std::vector<FI>& nodes,
    std::vector<std::shared_ptr<std::vector<int>>>& outPre)
{
    const int W32 = bmmaW32;
    const int n = (int)nodes.size();
    int kmax = 1;
    for (auto& x : nodes) kmax = std::max(kmax, (int)x.itemsSet.size());

    static thread_local unsigned* tl_dA = nullptr;
    static thread_local int* tl_dC = nullptr;
    static thread_local int* tl_dItems = nullptr;
    static thread_local int tl_rowsCap = 0;
    static thread_local int tl_kSplitMax = 1;
    if (tl_dA == nullptr) {
        int rows = (int)std::min<size_t>(1024, std::max<size_t>(64, BMMA_ABYTES_CAP / ((size_t)W32 * 4)));
        rows = (rows + 63) / 64 * 64;   // pad-row grid reads whole 64-row tiles: round capacity UP
        if (envWaveCap() > 0) rows = envWaveCap();   // ablation: fixed wave size
        tl_rowsCap = rows;
        // split-k budget: enough blocks to fill ~4 waves of the detected SMs; only for long-k geometry
        const long capBlocksXY = (long)(rows / 64) * (bmmaDpad / 64);
        tl_kSplitMax = (W32 >= 32768 && !envSplitkOff())
            ? (int)std::min<long>(32, std::max<long>(1, (4L * gpuSMCount()) / std::max<long>(1, capBlocksXY))) : 1;
        if (cudaMalloc(&tl_dA, (size_t)rows * W32 * 4) != cudaSuccess) { std::cerr << "dA alloc fail" << std::endl; exit(1); }
        if (cudaMalloc(&tl_dC, (size_t)rows * bmmaDpad * 4 * tl_kSplitMax) != cudaSuccess) { std::cerr << "dC alloc fail" << std::endl; exit(1); }
        if (cudaMalloc(&tl_dItems, (size_t)rows * 32 * 4) != cudaSuccess) { std::cerr << "dItems alloc fail" << std::endl; exit(1); }
    }
    if (kmax > 31) { std::cerr << "itemset depth exceeds batch packer" << std::endl; exit(1); }

    static thread_local std::vector<int> h_items;
    static thread_local std::vector<int> h_C;
    h_items.resize((size_t)tl_rowsCap * kmax);
    h_C.resize((size_t)tl_rowsCap * bmmaDpad);

    const size_t bytesB = (size_t)bmmaDpad * W32 * 4;
    outPre.resize(n);

    for (int base = 0; base < n; base += tl_rowsCap) {
        const int bcur = std::min(tl_rowsCap, n - base);
        for (int i = 0; i < bcur; i++) {
            int* dst = &h_items[(size_t)i * kmax];
            const FI& nd = nodes[base + i];
            for (int t = 0; t < (int)nd.itemsSet.size(); t++) dst[t] = nd.itemsSet[t];
            for (int t = (int)nd.itemsSet.size(); t < kmax; t++) dst[t] = nd.itemsSet[0];  // idempotent pad
        }
        cudaMemcpy(tl_dItems, h_items.data(), (size_t)bcur * kmax * 4, cudaMemcpyHostToDevice);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
        cudaEventRecord(e0);
        {
            const int W64 = W32 / 2;   // W32 is a multiple of 64 (2048-bit row alignment)
            const int wpb = envMaskWpb();
            dim3 mgrid(bcur, (W64 + wpb - 1) / wpb);
            maskBuildKernelFast <<<mgrid, 256 >>> ((unsigned long long*)tl_dA,
                (const unsigned long long*)d_bitmap, tl_dItems, kmax, W64, wpb);
        }
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); g_tMask += ms;

        const long blocks = (long)((bcur + 63) / 64) * (bmmaDpad / 64);
        // long-k geometry (N >= 1M bits) always prefers the smem-tiled kernel: b1 fragment
        // loads from global memory compile to scattered gathers; smem staging fixes that
        // regardless of B size. Short-k with few blocks stays on the naive kernel.
        // Long-k additionally splits k across grid.z when the (x,y) grid cannot fill the SMs.
        const bool longK = (W32 >= 32768);
        const bool useSmem = longK || ((blocks >= 2L * gpuSMCount()) && (bytesB > (size_t)256 * 1024 * 1024));
        const int kSplits = longK ? tl_kSplitMax : 1;
        const long splitStride = (long)tl_rowsCap * bmmaDpad;
        dim3 grid((bcur + 63) / 64, bmmaDpad / 64, kSplits);
        cudaEventRecord(e0);
        if (longK) {
            bmmaKernelSmemSplitK <<<grid, 256 >>> (tl_dA, d_bmmaB, tl_dC, bcur, bmmaDpad, W32, kSplits, splitStride);
            if (kSplits > 1) {
                const long rn = (long)bcur * bmmaDpad;
                bmmaSplitReduceKernel <<<(unsigned)((rn + 255) / 256), 256 >>> (tl_dC, kSplits, splitStride, rn);
            }
        }
        else if (useSmem) bmmaKernelSmem <<<grid, 256 >>> (tl_dA, d_bmmaB, tl_dC, bcur, bmmaDpad, W32);
        else              bmmaKernelNaive <<<grid, 256 >>> (tl_dA, d_bmmaB, tl_dC, bcur, bmmaDpad, W32);
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); g_tBmma += ms;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) { std::cerr << "BMMA launch fail: " << cudaGetErrorString(err) << std::endl; exit(1); }
        double cp0 = (double)clock() / CLOCKS_PER_SEC * 1000.0;
        cudaMemcpy(h_C.data(), tl_dC, (size_t)bcur * bmmaDpad * 4, cudaMemcpyDeviceToHost);
        g_tCopy += (double)clock() / CLOCKS_PER_SEC * 1000.0 - cp0;
        cudaEventDestroy(e0); cudaEventDestroy(e1);

        double sc0 = (double)clock() / CLOCKS_PER_SEC * 1000.0;
        for (int i = 0; i < bcur; i++) {
            auto pre = std::make_shared<std::vector<int>>(itemIDmax + 2, 0);
            int cnt = 0;
            const int* row = &h_C[(size_t)i * bmmaDpad];
            for (int c = 0; c < freqItemsNum; c++) {
                int v = row[c];
                pre->at(freqItems[c]) = v;
                if (v >= frequencyThreshold) cnt++;
            }
            pre->at(itemIDmax + 1) = cnt;
            outPre[base + i] = pre;
        }
        g_tScatter += (double)clock() / CLOCKS_PER_SEC * 1000.0 - sc0;
        g_bmmaBatches++; g_bmmaRows += bcur; if (useSmem) g_bmmaSmemBatches++;
    }
}

// expand one node whose resFrequency is already known: MFI check + push children
static void expandNode(FI& node, std::vector<int>& resFrequency)
{
    if (resFrequency[itemIDmax + 1] == node.itemsSet.size()) {
        MFIsPoolPush(&node);
        return;
    }
    for (int i = 0; i <= node.progress - 1; i++) {  // little-endian mode
        if (resFrequency[i] >= frequencyThreshold && isOutside(i, &node)) {
            FI superOne = node;
            superOne.itemsSet.push_back(i);
            superOne.frequency = resFrequency[i];
            superOne.progress = i;
            FIsStackkPush(&superOne);
        }
    }
}

// Wave engine: drain the task stack in waves, count the whole wave with one batched BMMA
// call (memo: "collect b candidates from the same search frontier"), then expand on CPU.
// NOTE: always BMMA, even for tiny waves -- padding to a 64-row tile costs ~1 ms while one
// per-node GPU call costs tens of ms at large N, so the per-node fallback never pays off here.
static void runWaveEngine()
{
    const int waveCap = envWaveCap() > 0 ? envWaveCap()
        : (int)std::min<size_t>(1024, std::max<size_t>(64, BMMA_ABYTES_CAP / ((size_t)bmmaW32 * 4))) / 64 * 64;
    FI one;
    while (true) {
        std::vector<FI> wave;
        while ((int)wave.size() < waveCap && FIsStackkPop(&one)) wave.push_back(one);
        if (wave.empty()) break;
        if ((int)wave.size() < BMMA_BATCH_MIN) g_bmmaFallbackWaves++;   // small wave, padded (stat only)
        std::vector<std::shared_ptr<std::vector<int>>> pres;
        batchCountBMMA(wave, pres);
        for (size_t i = 0; i < wave.size(); i++) expandNode(wave[i], *pres[i]);
    }
}

// ---- L1 router execution paths ----
// SIMT wave counting: per-node counting dispatched across CPU worker threads,
// matching the tuned baseline's dominant path (RUN_MODE=2 per-node dispatcher).
// ROUTE_SIMT=gpu switches to the per-node CUDA kernel instead (diagnostic path).
// Output layout is identical to batchCountBMMA's (freqItems positions + count slot),
// which is what keeps results bit-identical across routing histories.
static bool envSimtGpu() { const char* s = getenv("ROUTE_SIMT"); return s && *s && strcmp(s, "gpu") == 0; }
static void countWaveSimt(const std::vector<FI>& wave,
    std::vector<std::shared_ptr<std::vector<int>>>& outPre)
{
    const int w = (int)wave.size();
    outPre.resize(w);
    const bool useGpu = envSimtGpu();
    Concurrency::parallel_for(0, w, [&](int i) {
        std::vector<int> res(itemIDmax + 2, 0);
        if (useGpu) {
            if (getResFrequencyCuda(wave[i], &res) != cudaSuccess) { std::cerr << "SIMT count fail" << std::endl; exit(1); }
        } else {
            getResFrequency(wave[i], &res);
        }
        outPre[i] = std::make_shared<std::vector<int>>(std::move(res));
    });
}

static int waveKmax(const std::vector<FI>& wave) {
    int kmax = 1;
    for (auto& x : wave) kmax = std::max(kmax, (int)x.itemsSet.size());
    return kmax;
}

// Staged tensor path for the Mode-E pipeline: double-buffered A/C, two streams.
// mask(i+1) on sMask overlaps BMMA(i) on sBmma (Thm. 3 steady state).
struct TensorPipe {
    unsigned* dA[2] = { nullptr, nullptr };
    int* dC[2] = { nullptr, nullptr };
    int* dItems[2] = { nullptr, nullptr };
    cudaStream_t sMask = nullptr, sBmma = nullptr;
    cudaEvent_t evMask[2] = { nullptr, nullptr }, evBmma[2] = { nullptr, nullptr };
    int rowsCap = 0, kSplitMax = 1, kmaxCap = 32;
    long splitStride = 0;
    std::vector<int> h_items, h_C;

    void init() {
        const int W32 = bmmaW32;
        int rows = (int)std::min<size_t>(1024, std::max<size_t>(64, BMMA_ABYTES_CAP / ((size_t)W32 * 4)));
        rows = (rows + 63) / 64 * 64;
        if (envWaveCap() > 0) rows = envWaveCap();
        rowsCap = rows;
        const long capBlocksXY = (long)(rows / 64) * (bmmaDpad / 64);
        kSplitMax = (W32 >= 32768 && !envSplitkOff())
            ? (int)std::min<long>(32, std::max<long>(1, (4L * gpuSMCount()) / std::max<long>(1, capBlocksXY))) : 1;
        splitStride = (long)rowsCap * bmmaDpad;
        for (int p = 0; p < 2; p++) {
            if (cudaMalloc(&dA[p], (size_t)rowsCap * W32 * 4) != cudaSuccess) { std::cerr << "pipe dA alloc fail" << std::endl; exit(1); }
            if (cudaMalloc(&dC[p], (size_t)rowsCap * bmmaDpad * 4 * kSplitMax) != cudaSuccess) { std::cerr << "pipe dC alloc fail" << std::endl; exit(1); }
            if (cudaMalloc(&dItems[p], (size_t)rowsCap * 32 * 4) != cudaSuccess) { std::cerr << "pipe dItems alloc fail" << std::endl; exit(1); }
            cudaEventCreateWithFlags(&evMask[p], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&evBmma[p], cudaEventDisableTiming);
        }
        cudaStreamCreateWithFlags(&sMask, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&sBmma, cudaStreamNonBlocking);
    }
    void destroy() {
        for (int p = 0; p < 2; p++) {
            if (evMask[p]) cudaEventDestroy(evMask[p]);
            if (evBmma[p]) cudaEventDestroy(evBmma[p]);
        }
        if (sMask) cudaStreamDestroy(sMask);
        if (sBmma) cudaStreamDestroy(sBmma);
    }
};

// pack item ids + H2D + mask build on sMask; record evMask[buf]
static void pipeStageMask(TensorPipe& P, const std::vector<FI>& wave, int kmax, int buf) {
    const int W32 = bmmaW32;
    const int bcur = (int)wave.size();
    P.h_items.resize((size_t)P.rowsCap * kmax);
    for (int i = 0; i < bcur; i++) {
        int* dst = &P.h_items[(size_t)i * kmax];
        const FI& nd = wave[i];
        for (int t = 0; t < (int)nd.itemsSet.size(); t++) dst[t] = nd.itemsSet[t];
        for (int t = (int)nd.itemsSet.size(); t < kmax; t++) dst[t] = nd.itemsSet[0];
    }
    cudaMemcpyAsync(P.dItems[buf], P.h_items.data(), (size_t)bcur * kmax * 4, cudaMemcpyHostToDevice, P.sMask);
    const int W64 = W32 / 2;
    const int wpb = envMaskWpb();
    dim3 mgrid(bcur, (W64 + wpb - 1) / wpb);
    maskBuildKernelFast <<<mgrid, 256, 0, P.sMask >>> ((unsigned long long*)P.dA[buf],
        (const unsigned long long*)d_bitmap, P.dItems[buf], kmax, W64, wpb);
    cudaEventRecord(P.evMask[buf], P.sMask);
}

// BMMA on sBmma (waits mask event); record evBmma[buf]
static void pipeStageBmma(TensorPipe& P, int bcur, int buf) {
    const int W32 = bmmaW32;
    const long blocks = (long)((bcur + 63) / 64) * (bmmaDpad / 64);
    const bool longK = (W32 >= 32768);
    const bool useSmem = longK || ((blocks >= 2L * gpuSMCount()) && ((size_t)bmmaDpad * W32 * 4 > (size_t)256 * 1024 * 1024));
    const int kSplits = longK ? P.kSplitMax : 1;
    dim3 grid((bcur + 63) / 64, bmmaDpad / 64, kSplits);
    cudaStreamWaitEvent(P.sBmma, P.evMask[buf], 0);
    if (longK) {
        bmmaKernelSmemSplitK <<<grid, 256, 0, P.sBmma >>> (P.dA[buf], d_bmmaB, P.dC[buf], bcur, bmmaDpad, W32, kSplits, P.splitStride);
        if (kSplits > 1) {
            const long rn = (long)bcur * bmmaDpad;
            bmmaSplitReduceKernel <<<(unsigned)((rn + 255) / 256), 256, 0, P.sBmma >>> (P.dC[buf], kSplits, P.splitStride, rn);
        }
    }
    else if (useSmem) bmmaKernelSmem <<<grid, 256, 0, P.sBmma >>> (P.dA[buf], d_bmmaB, P.dC[buf], bcur, bmmaDpad, W32);
    else              bmmaKernelNaive <<<grid, 256, 0, P.sBmma >>> (P.dA[buf], d_bmmaB, P.dC[buf], bcur, bmmaDpad, W32);
    cudaEventRecord(P.evBmma[buf], P.sBmma);
}

// wait BMMA[buf], read back C, scatter to per-node resFrequency (identical layout to SIMT path)
static void pipeReadback(TensorPipe& P, const std::vector<FI>& wave, int buf,
    std::vector<std::shared_ptr<std::vector<int>>>& outPre)
{
    const int bcur = (int)wave.size();
    cudaEventSynchronize(P.evBmma[buf]);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "pipe BMMA fail: " << cudaGetErrorString(err) << std::endl; exit(1); }
    P.h_C.resize((size_t)P.rowsCap * bmmaDpad);
    cudaMemcpy(P.h_C.data(), P.dC[buf], (size_t)bcur * bmmaDpad * 4, cudaMemcpyDeviceToHost);
    outPre.resize(bcur);
    for (int i = 0; i < bcur; i++) {
        auto pre = std::make_shared<std::vector<int>>(itemIDmax + 2, 0);
        int cnt = 0;
        const int* row = &P.h_C[(size_t)i * bmmaDpad];
        for (int c = 0; c < freqItemsNum; c++) {
            int v = row[c];
            pre->at(freqItems[c]) = v;
            if (v >= frequencyThreshold) cnt++;
        }
        pre->at(itemIDmax + 1) = cnt;
        outPre[i] = pre;
    }
}

// Mode D / R: routed execution, wave-by-wave, synchronous.
static void runRouterEngineSimple() {
    const int waveCap = envWaveCap() > 0 ? envWaveCap()
        : (int)std::min<size_t>(1024, std::max<size_t>(64, BMMA_ABYTES_CAP / ((size_t)bmmaW32 * 4))) / 64 * 64;
    const bool oracle = envOracle() != 0;
    FI one;
    int lastDecision = -1;
    long waveIdx = 0;
    while (true) {
        auto c0 = std::chrono::high_resolution_clock::now();
        std::vector<FI> wave;
        while ((int)wave.size() < waveCap && FIsStackkPop(&one)) wave.push_back(one);
        g_tCollect += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - c0).count() * 1000.0;
        if (wave.empty()) break;
        const int w = (int)wave.size();
        const int kmax = waveKmax(wave);
        auto d0 = std::chrono::high_resolution_clock::now();
        int d = routeDecision(w, kmax, lastDecision);
        auto d1 = std::chrono::high_resolution_clock::now();
        double decideUs = std::chrono::duration<double>(d1 - d0).count() * 1e6;
        lastDecision = d;
        double estS = estCost(w, g_rateSimt, g_rateSimtAll);
        double estB = estCost(w, g_rateBmma, g_rateBmmaAll);
        std::vector<std::shared_ptr<std::vector<int>>> pres;
        double oS = -1, oB = -1, actualMs = 0;
        auto e0 = std::chrono::high_resolution_clock::now();
        if (d == 1) {
            g_wavesTensor++; if (w < BMMA_BATCH_MIN) g_wavesTensorSmall++;
            if (oracle) { auto t = std::chrono::high_resolution_clock::now(); std::vector<std::shared_ptr<std::vector<int>>> tmp; countWaveSimt(wave, tmp); oS = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count() * 1000.0; }
            batchCountBMMA(wave, pres);
        } else {
            g_wavesSimt++;
            countWaveSimt(wave, pres);
            if (oracle) { auto t = std::chrono::high_resolution_clock::now(); std::vector<std::shared_ptr<std::vector<int>>> tmp; batchCountBMMA(wave, tmp); oB = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t).count() * 1000.0; }
        }
        actualMs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - e0).count() * 1000.0;
        bool warmup = routeLearn(d, w, actualMs);
        routeLogLine(waveIdx++, w, kmax, estS, estB, d, actualMs, decideUs, warmup, oS, oB);
        auto x0 = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < wave.size(); i++) expandNode(wave[i], *pres[i]);
        g_tExpand += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - x0).count() * 1000.0;
    }
}

// Mode E: routed execution + two-stream overlap (mask of wave i+1 behind BMMA of wave i).
struct PendingWave {
    std::vector<FI> wave;
    long idx = 0; int w = 0, kmax = 0;
    double estS = -1, estB = -1, decideUs = 0;
    std::chrono::high_resolution_clock::time_point stageT;
};
static void runRouterEnginePipeline() {
    const int waveCap = envWaveCap() > 0 ? envWaveCap()
        : (int)std::min<size_t>(1024, std::max<size_t>(64, BMMA_ABYTES_CAP / ((size_t)bmmaW32 * 4))) / 64 * 64;
    TensorPipe P; P.init();
    FI one;
    int lastDecision = -1;
    long waveIdx = 0;
    int buf = 0;
    bool inflight = false;
    PendingWave cur;
    while (true) {
        std::vector<FI> wave;
        while ((int)wave.size() < waveCap && FIsStackkPop(&one)) wave.push_back(one);
        if (wave.empty()) {
            if (!inflight) break;
            // drain the in-flight tensor wave: its expansion may refill the stack
            std::vector<std::shared_ptr<std::vector<int>>> pres;
            pipeReadback(P, cur.wave, buf ^ 1, pres);
            double actualMs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cur.stageT).count() * 1000.0;
            bool warmup = routeLearn(1, cur.w, actualMs);
            routeLogLine(cur.idx, cur.w, cur.kmax, cur.estS, cur.estB, 1, actualMs, cur.decideUs, warmup);
            for (size_t i = 0; i < cur.wave.size(); i++) expandNode(cur.wave[i], *pres[i]);
            inflight = false;
            continue;
        }
        const int w = (int)wave.size();
        const int kmax = waveKmax(wave);
        if (kmax > 31) { std::cerr << "itemset depth exceeds pipe packer" << std::endl; exit(1); }
        auto d0 = std::chrono::high_resolution_clock::now();
        int d = routeDecision(w, kmax, lastDecision);
        auto d1 = std::chrono::high_resolution_clock::now();
        double decideUs = std::chrono::duration<double>(d1 - d0).count() * 1e6;
        lastDecision = d;
        double estS = estCost(w, g_rateSimt, g_rateSimtAll);
        double estB = estCost(w, g_rateBmma, g_rateBmmaAll);

        if (d == 1) {
            g_wavesTensor++; if (w < BMMA_BATCH_MIN) g_wavesTensorSmall++;
            // tensor wave: stage mask now (overlaps in-flight BMMA of the previous wave)
            cur.stageT = std::chrono::high_resolution_clock::now();
            pipeStageMask(P, wave, kmax, buf);
            if (inflight) {
                // read back the previous wave, then IMMEDIATELY launch its successor's
                // BMMA so the host-side expansion below overlaps tensor execution
                std::vector<std::shared_ptr<std::vector<int>>> pres;
                pipeReadback(P, cur.wave, buf ^ 1, pres);
                double actualMs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cur.stageT).count() * 1000.0;
                bool warmup = routeLearn(1, cur.w, actualMs);
                routeLogLine(cur.idx, cur.w, cur.kmax, cur.estS, cur.estB, 1, actualMs, cur.decideUs, warmup);
                pipeStageBmma(P, w, buf);          // keep the tensor engine busy first
                for (size_t i = 0; i < cur.wave.size(); i++) expandNode(cur.wave[i], *pres[i]);
            } else {
                pipeStageBmma(P, w, buf);
            }
            cur.wave = wave; cur.idx = waveIdx++; cur.w = w; cur.kmax = kmax;
            cur.estS = estS; cur.estB = estB; cur.decideUs = decideUs;
            inflight = true;
            buf ^= 1;
        } else {
            g_wavesSimt++;
            // SIMT wave: per-node counting runs on its own stream, overlaps in-flight BMMA
            std::vector<std::shared_ptr<std::vector<int>>> pres;
            auto s0 = std::chrono::high_resolution_clock::now();
            countWaveSimt(wave, pres);
            double actualMs = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - s0).count() * 1000.0;
            bool warmup = routeLearn(0, w, actualMs);
            routeLogLine(waveIdx++, w, kmax, estS, estB, 0, actualMs, decideUs, warmup);
            for (size_t i = 0; i < wave.size(); i++) expandNode(wave[i], *pres[i]);
        }
    }
    P.destroy();
}

// Startup calibration probe (Sec. 5.3): time both engines on a synthetic wave
// built from the real bitmap so the cost table starts populated — the first real
// wave routes by measurement, not by prior. Counts on the probe are discarded;
// only wall time enters the table. The tensor probe runs twice: the first call
// pays one-time cudaMalloc/module-init and is discarded as warmup.
static void routeCalibrate() {
    if (freqItemsNum < 1) return;
    const int PW = std::min(256, std::max(64, freqItemsNum));
    const int ia = 0, ib = freqItemsNum > 1 ? 1 : 0;
    std::vector<FI> probe(PW);
    for (int i = 0; i < PW; i++) {
        probe[i].itemsSet.push_back(ia);
        if (ib) probe[i].itemsSet.push_back(ib);
        probe[i].root = ia; probe[i].progress = 1; probe[i].frequency = 0;
    }
    {   // SIMT probe: parallel per-node counting (genuine rate)
        std::vector<std::shared_ptr<std::vector<int>>> tmp;
        auto t0 = std::chrono::high_resolution_clock::now();
        countWaveSimt(probe, tmp);
        double ms = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() * 1000.0;
        fprintf(stderr, "[router] probe simt(256): %.4f ms\n", ms);
        routeLearn(0, PW, ms);
    }
    {   // tensor probe: first pass = init warmup, second pass = genuine rate
        std::vector<std::shared_ptr<std::vector<int>>> tmp;
        auto tA = std::chrono::high_resolution_clock::now();
        batchCountBMMA(probe, tmp);
        double ms1 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tA).count() * 1000.0;
        g_warmBmma = true;   // the probe absorbed cudaMalloc/module init
        auto t0 = std::chrono::high_resolution_clock::now();
        batchCountBMMA(probe, tmp);
        double ms = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() * 1000.0;
        fprintf(stderr, "[router] probe tensor(256): warmup=%.4f ms genuine=%.4f ms\n", ms1, ms);
        routeLearn(1, PW, ms);
    }
}

static void runRouterEngine() {
    routeLogOpen();
    const char* np = getenv("ROUTE_PROBE");
    if (envRouteMode() != 3 && np && *np && atoi(np) != 0) routeCalibrate();   // off by default: probe costs more than it saves on small datasets
    if (envRouteMode() == 2) runRouterEnginePipeline();
    else runRouterEngineSimple();
    fprintf(stderr, "[router] simt_waves=%ld tensor_waves=%ld (small=%ld) tDecide=%.4f ms tCollect=%.4f ms tExpand=%.4f ms\n",
        g_wavesSimt, g_wavesTensor, g_wavesTensorSmall, g_tDecide, g_tCollect, g_tExpand);
}

static void recursiveExpansion(FI newOne) {
    std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);
    //getResFrequency(newOne, &resFrequency);
    getResFrequencyCuda(newOne, &resFrequency);

    if (resFrequency[itemIDmax + 1] == newOne.itemsSet.size()) {
        MFIsPoolPush(&newOne);
    }
    else
    {
        //for (int i = newOne.progress + 1; i <= itemIDmax; i++)//Big-endian mode 
        for (int i = 0; i <= newOne.progress - 1; i++)//little-endian mode
        {
            if (resFrequency[i] >= frequencyThreshold)
            {
                if (isOutside(i, &newOne)) {
                    FI superOne = newOne;
                    superOne.itemsSet.push_back(i);
                    superOne.frequency = resFrequency[i];
                    superOne.progress = i;
                    recursiveExpansion(superOne);
                }
            }
        }
    }
}

static void displayTransSetInformation() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "transSetFile: " << transSetFile << std::endl;
    std::cout << "dimension: " << dimension << std::endl;
    std::cout << "transNum: " << transNum << std::endl;
    std::cout << "itemIDmin: " << itemIDmin << std::endl;
    std::cout << "itemIDmax: " << itemIDmax << std::endl;
    std::cout << "transLengthMin: " << transLengthMin << std::endl;
    std::cout << "transLengthMax: " << transLengthMax << std::endl;
    std::cout << "transLengthMean: " << transLengthMean << std::endl;

    std::cout << std::endl << "supportThreshold: " << supportThreshold << std::endl << std::endl;

    std::cout << "dimensionReduced: " << dimensionReduced << std::endl;
    std::cout << "transNumReduced: " << transNumReduced << std::endl;
    std::cout << "transLengthReducedMin: " << transLengthReducedMin << std::endl;
    std::cout << "transLengthReducedMax: " << transLengthReducedMax << std::endl;
    std::cout << "transLengthReducedMean: " << transLengthReducedMean << std::endl;
}

static void displayMFIsPool() {
    std::cout << "============================List of MFIs:=============================" << std::endl;
    for (int i = 0; i < MFIsPool.size(); i++) {
        std::cout << i + 1 << "th: " << MFIsPool[i].itemsSet.size() << " { ";
        for (int j = 0; j < MFIsPool[i].itemsSet.size(); j++) {
            std::cout << denseToOrig[MFIsPool[i].itemsSet[j]] << " ";
        }
        std::cout << "}   support:" << double(MFIsPool[i].frequency) / transNum << "  frequency: " << MFIsPool[i].frequency << std::endl;
    }
}

static void output2displayer() {
    displayMFIsPool();
    std::cout << "======================================================================" << std::endl;
    std::cout << "preprocessing_time(s): " << preprocessing_time << std::endl;
    std::cout << "computation_time(s): " << computation_time << std::endl;
    std::cout << "total_time(s): " << preprocessing_time + computation_time << std::endl;

    std::cout << std::endl << "MFIsNumber: " << MFIsPool.size() << std::endl;
    if (envWaveCap() > 0 || envSplitkOff() || getenv("BMMA_MASK_WPB"))
        std::cout << "ABLATION waveCap=" << (envWaveCap() > 0 ? envWaveCap() : -1)
                  << " splitk=" << (envSplitkOff() ? "OFF" : "auto")
                  << " mask_wpb=" << envMaskWpb() << std::endl;
    std::cout << "BMMA batches: " << g_bmmaBatches << " (smem: " << g_bmmaSmemBatches
              << "), rows counted by BMMA: " << g_bmmaRows
              << ", fallback waves: " << g_bmmaFallbackWaves << std::endl;
    std::cout << "BMMA phase ms: mask=" << g_tMask << " bmma=" << g_tBmma
              << " memcpy=" << g_tCopy << " scatter=" << g_tScatter << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl << std::endl;
}

static void outputWorkingEnvironment2file(std::ofstream& outfile) {
    if (outfile.is_open()) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount); // Get and display GPU information
        for (int i = 0; i < deviceCount; ++i)
        {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            outfile << "GPU " << i << ": " << deviceProp.name << std::endl;
            outfile << "     Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
            outfile << "     Max Threads Per Multi-Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
            outfile << "     Warp Size: " << deviceProp.warpSize << std::endl;
            outfile << "     Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
            outfile << "     Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
            outfile << "     Number of Multi-Processors: " << deviceProp.multiProcessorCount << std::endl;
            int coresPerSM = getCoresPerSM(deviceProp.major, deviceProp.minor);
            if (coresPerSM == -1) {
                outfile << "Unknown architecture, unable to calculate the number of CUDA cores !" << std::endl;
            }
            else {
                int totalCores = coresPerSM * deviceProp.multiProcessorCount;
                outfile << "     Number of CUDA Cores Per SM: " << coresPerSM << std::endl;
                outfile << "     Total Number of CUDA Cores: " << totalCores << std::endl;
            }
            outfile << std::endl;
        }

        CString cpuName;
        getCPUInfo(cpuName, CPUphysicalCores, CPUlogicalCores);
        outfile << "CPU: " << cpuName << std::endl;
        {
            outfile << "     Number of Physical Cores: " << CPUphysicalCores << std::endl;
            outfile << "     Number of Logical Cores: " << CPUlogicalCores << std::endl << std::endl;
        }

        outfile << "Visual Studio: " << getVSversion() << ", MSC_VER: " << _MSC_VER << std::endl;
        int rt_ver = 0;
        cudaError_t err = cudaRuntimeGetVersion(&rt_ver);
        if (err == cudaSuccess) outfile << "CUDA Version: " << rt_ver / 1000 << "." << (rt_ver % 1000) / 10 << std::endl;
        else outfile << "CUDA Version not detected: " << cudaGetErrorString(err) << std::endl;
    }
}

static void output2file(CString resultFile, int dimensionReduced) {
    std::ofstream outfile(resultFile);
    outfile << "==================================================================" << std::endl;
    outputWorkingEnvironment2file(outfile);
    outfile << "==================================================================" << std::endl;
    outfile << "transSetFile: " << transSetFile << std::endl;
    outfile << "dimension: " << dimension << std::endl;
    outfile << "transNum: " << transNum << std::endl;
    outfile << "itemIDmin: " << itemIDmin << std::endl;
    outfile << "itemIDmax: " << itemIDmax << std::endl;
    outfile << "transLengthMin: " << transLengthMin << std::endl;
    outfile << "transLengthMax: " << transLengthMax << std::endl;
    outfile << "transLengthMean: " << transLengthMean << std::endl;

    outfile << std::endl << "supportThreshold: " << supportThreshold << std::endl << std::endl;

    outfile << "dimensionReduced: " << dimensionReduced << std::endl;
    outfile << "transNumReduced: " << transNumReduced << std::endl;
    outfile << "transLengthReducedMin: " << transLengthReducedMin << std::endl;
    outfile << "transLengthReducedMax: " << transLengthReducedMax << std::endl;
    outfile << "transLengthReducedMean: " << transLengthReducedMean << std::endl;

    outfile << std::endl << "GPUtaskPercentage: " << GPUtaskPercentage << "%" << std::endl;
    outfile << "==================================================================" << std::endl;
    outfile << "preprocessing_time(s): " << preprocessing_time << std::endl;
    outfile << "computation_time(s): " << computation_time << std::endl;
    outfile << "total_time(s): " << preprocessing_time + computation_time << std::endl;

    outfile << std::endl << "MFIsNumber: " << MFIsPool.size() << std::endl;
    outfile << "============================List of MFIs:=============================" << std::endl;
    for (int i = 0; i < MFIsPool.size(); i++) {
        outfile << i + 1 << "th: " << MFIsPool[i].itemsSet.size() << " { ";
        for (int j = 0; j < MFIsPool[i].itemsSet.size(); j++) {
            outfile << denseToOrig[MFIsPool[i].itemsSet[j]] << " ";
        }
        outfile << "}   support: " << double(MFIsPool[i].frequency) / transNum << "  frequency: " << MFIsPool[i].frequency << std::endl;
    }
    outfile << "==================================================================" << std::endl;
    outfile.close();
}

static void reduceTransSet() {//精减事务集
    for (auto& row : h_transSet) {
        row.erase(std::remove_if(row.begin(), row.end(),
            [&](int x) { return (x >= 0 && x < freqPerItem.size()) && (freqPerItem[x] < frequencyThreshold); }),
            row.end());
    }
    // 两遍法压缩：一次性删除全部空行，避免逐行 erase 触发 O(N^2) 搬移（kosarak 上 55s->~1s）
    h_transSet.erase(std::remove_if(h_transSet.begin(), h_transSet.end(),
        [](const std::vector<int>& r) { return r.empty(); }),
        h_transSet.end());
    //for (const auto& r : h_transSet) { for (int val : r) std::cout << val << ' '; std::cout << std::endl; }// 输出结果
}

void create_d_transSet(int max_transLengthReduced, int transNumReduced)
{
    // 1. 参数合法性校验
    if (max_transLengthReduced <= 0 || transNumReduced <= 0) {
        std::cout << "参数错误：max_transLengthReduced 或 transNumReduced 不能为非正数" << std::endl;
        return;
    }
    cudaError_t cudaStatus = cudaSuccess;
    // 2. 释放旧内存（避免内存泄漏）
    if (d_transSet != nullptr) {
        cudaFree(d_transSet);
        d_transSet = nullptr;
    }
    // 3. 分配设备端连续内存
    size_t totalSize = transNumReduced * (max_transLengthReduced + 1) * sizeof(int);
    cudaStatus = cudaMalloc(&d_transSet, totalSize);
    if (cudaStatus != cudaSuccess) {
        std::cout << "设备端d_transSet连续内存分配失败: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    // 4. 主机端一次性拼装整块连续缓冲，再一次 cudaMemcpy 完成传输
    //    （原先逐行拷贝，accidents 约 30 万行即 30 万次 API 调用，耗时数秒）
    size_t rowPitch = (size_t)max_transLengthReduced + 1;
    int* h_block = new (std::nothrow) int[(size_t)transNumReduced * rowPitch];
    if (h_block == nullptr) {
        std::cout << "主机端h_block内存分配失败" << std::endl;
        cudaFree(d_transSet);
        d_transSet = nullptr;
        return;
    }
    for (int i = 0; i < transNumReduced; ++i) {
        int* dst = h_block + (size_t)i * rowPitch;
        dst[0] = static_cast<int>(h_transSet[i].size());
        for (size_t j = 0; j < h_transSet[i].size(); j++) {
            dst[j + 1] = h_transSet[i][j];
        }
    }
    cudaStatus = cudaMemcpy(d_transSet, h_block, totalSize, cudaMemcpyHostToDevice);
    delete[] h_block;
    if (cudaStatus != cudaSuccess) {
        std::cout << "设备事务集数据整体拷贝失败: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_transSet);
        d_transSet = nullptr;
        return;
    }
    //std::cout << std::endl << "The transaction set on the device side has been created successfully !" << std::endl;
}

static double getGPUtaskPercentage(int parallelNum)
{
    std::vector<double> CPU_test_duration(parallelNum + 1, 0.0);
    std::vector<double> GPU_test_duration(parallelNum + 1, 0.0);

    Concurrency::parallel_for
    (1, parallelNum + 1, [&](int parallelID)
        {
            FI newOne;
            newOne.itemsSet.push_back(items_inFreqSort[0]);
            newOne.frequency = freqPerItem_inSort[0];
            newOne.root = items_inFreqSort[0];
            newOne.progress = items_inFreqSort[0];

            std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);
            int times = 10;

            auto t3 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < times; i++)  getResFrequencyCuda(newOne, &resFrequency);
            auto t4 = std::chrono::high_resolution_clock::now();
            float GPU_time = std::chrono::duration<float, std::milli>(t4 - t3).count();

            auto t1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < times; i++)  getResFrequency(newOne, &resFrequency);
            auto t2 = std::chrono::high_resolution_clock::now();
            float CPU_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

            CPU_test_duration[parallelID] = CPU_time;
            GPU_test_duration[parallelID] = GPU_time;
        }
    );

    for (int i = 1; i < parallelNum + 1; i++)
    {
        CPU_test_duration[0] += CPU_test_duration[i];
        GPU_test_duration[0] += GPU_test_duration[i];
    }

    double GPUpercentage = 100 * CPU_test_duration[0] / (CPU_test_duration[0] + GPU_test_duration[0]);
    std::cout << std::endl << "CPU_test_duration: " << CPU_test_duration[0] << "; GPU_test_duration: " << GPU_test_duration[0] << "; GPUpercentage: " << GPUpercentage << std::endl;
    return GPUpercentage;
}
//=====================================================================================================================================================================================================================================================
int main(int argc, char** argv)
{
    if (argc >= 3) { transSetFile = CString(argv[1]); supportThreshold = atof(argv[2]); }  // runtime override (regression/USB)
    displayWorkingEnvironment();
    //=================================================================================================================
    if (!readTransSetFile(transSetFile)) return 1;
    //=================================================================================================================
    getItemsFrequenc();
    //=================================================================================================================
    int CPU_parallel_num = (RUN_MODE == 0) ? 1 : CPUlogicalCores;// 串行模式单线程
    frequencyThreshold = transNum * supportThreshold;

    clock_t tic, toc;
    tic = clock();
    reduceTransSet();
    toc = clock();
    preprocessing_time = (double)(toc - tic) / CLOCKS_PER_SEC;

    transNumReduced = h_transSet.size();
    dimensionReduced = 0;
    for (int val : freqPerItem) if (double(val) / transNum > supportThreshold) dimensionReduced++;

    transLengthReducedMin = INT_MAX;
    transLengthReducedMax = 0;
    transLengthReducedMean = 0;
    for (const auto& row : h_transSet) {//获取最小最大长度
        if (row.size() < transLengthReducedMin) transLengthReducedMin = row.size();
        if (row.size() > transLengthReducedMax) transLengthReducedMax = row.size();
        transLengthReducedMean += row.size();
    }
    transLengthReducedMean /= transNumReduced;
    //=================================================================================================================
    // 构建位图（binary vector mapping）：每个项一个位向量，计时并入 preprocessing_time
    tic = clock();
    buildBitmap();
    toc = clock();
    preprocessing_time += (double)(toc - tic) / CLOCKS_PER_SEC;
    //=================================================================================================================
    displayTransSetInformation();
    //=================================================================================================================
    uploadBitmapToGPU();// 位图版：GPU 侧只需位图与幸存项列表，不再逐行传输事务集
    GPUtaskPercentage = getGPUtaskPercentage(CPU_parallel_num);
    //=================================================================================================================
    auto comp_t0 = std::chrono::high_resolution_clock::now();// 毫秒级以下耗时需高精度计时
    //=================================================================================================================
    for (int i = 0; i < dimensionReduced; i++) {
        FI newOne;
        newOne.itemsSet.push_back(items_inFreqSort[i]);
        newOne.frequency = freqPerItem_inSort[i];
        newOne.root = items_inFreqSort[i];
        newOne.progress = items_inFreqSort[i];

        //recursiveExpansion(newOne);
        FIsStackkPush(&newOne);
    }

    /**/
    if (RUN_MODE == 1 && BMMA_ENGINE) {
        if (envRouteMode() > 0) runRouterEngine();   // L1 co-scheduling router (Mode D/E/R)
        else runWaveEngine();                        // batched BMMA wave engine
    }
    else
    Concurrency::parallel_for//正文
    (1, CPU_parallel_num + 1, [](int parallelID)
        {
            FI popOne;
            std::vector<int> resFrequency(itemIDmax + 1 + 1, 0);

            while (true)
            {
                if (!FIsStackkPop(&popOne)) break;
                else
                {
                    if (RUN_MODE == 0) getResFrequency(popOne, &resFrequency);
                    else if (RUN_MODE == 1) getResFrequencyCuda(popOne, &resFrequency);
                    else { if ((rand() % 10000 + 1) <= GPUtaskPercentage * 100) getResFrequencyCuda(popOne, &resFrequency); else getResFrequency(popOne, &resFrequency); }
                    //getResFrequency(popOne, &resFrequency);
                    //getResFrequencyCuda(popOne, &resFrequency);

                    if (resFrequency[itemIDmax + 1] == popOne.itemsSet.size()) {
                        MFIsPoolPush(&popOne);
                    }
                    else
                    {
                        //for (int i = popOne.progress + 1; i <= itemIDmax; i++)//Big-endian mode 
                        for (int i = 0; i <= popOne.progress - 1; i++)//little-endian mode
                        {
                            //if (resFrequency[i] >= freqPerItem_inSort[m_generation])
                            if (resFrequency[i] >= frequencyThreshold)
                            {
                                if (isOutside(i, &popOne)) {
                                    FI superOne = popOne;
                                    superOne.itemsSet.push_back(i);
                                    superOne.frequency = resFrequency[i];
                                    superOne.progress = i;
                                    FIsStackkPush(&superOne);
                                }
                            }
                        }
                    }
                }
            }
        }
    );

    auto comp_t1 = std::chrono::high_resolution_clock::now();
    computation_time = std::chrono::duration<double>(comp_t1 - comp_t0).count();

    std::sort(MFIsPool.begin(), MFIsPool.end(),
        [](const auto& a, const auto& b) {
            return a.frequency > b.frequency; // 降序：frequency大的在前
        });

    output2displayer();

    CString resultFile;//生成结果文件 
    resultFile.Format(_T("%s-%f=Results.txt"), transSetFile, supportThreshold);
    output2file(resultFile, dimensionReduced);

    return 1;
}