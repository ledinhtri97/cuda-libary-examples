#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

inline int cudaDeviceInit(int argc, const char **argv) {
  int deviceCount;
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
    exit(EXIT_FAILURE);
  }

  int dev = findCudaDevice(argc, argv);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name
            << std::endl;

  checkCudaErrors(cudaSetDevice(dev));

  return dev;
}

bool printfNPPinfo(int argc, char *argv[]) {
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}



int main(int argc, char *argv[]) {
  printf("%s Starting...\n\n", argv[0]);

  try {
    std::string sFilename;
    char *filePath;

    cudaDeviceInit(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false) {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    } else {
      filePath = sdkFindFilePath("car.png", argv[0]);
    }

    if (filePath) {
      sFilename = filePath;
    } else {
      sFilename = "car.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good()) {
      std::cout << "wrapPerspectiveNPP opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    } else {
      std::cout << "wrapPerspectiveNPP unable to open: <" << sFilename.data()
                << ">" << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0) {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos) {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_carwrapPer.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }

    // declare a host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load gray-scale image from disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from the host image,
    // i.e. upload host to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

    NppiRect oSrcROI;
    oSrcROI.x = 0; oSrcROI.y = 0; oSrcROI.width = oSrcSize.width; oSrcROI.height = oSrcSize.height;
    NppiRect oDstROI;
    oDstROI.x = 0; oDstROI.y = 0; oDstROI.width = 240; oDstROI.height = 80;


    // allocate device image of appropriately reduced size
    npp::ImageNPP_8u_C1 oDeviceDst(oDstROI.width, oDstROI.height);

  /*
  nppiWarpPerspective_8u_C3R (const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, 
    int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation)
  */
  /*
  nppiWarpPerspective_8u_C1R (const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst, 
  int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation)
  */
    const double aCoeffs[3][3] = {
      {0.003738873871043,-0.000074685973232,-0.309865146875381},
      {0.000362480554031,0.004359462298453,-0.950762450695038},
      {-0.000000000024566,-0.000000000007490,0.001114780548960}
    };
    
    NPP_CHECK_NPP(nppiWarpPerspective_8u_C1R(
          oDeviceSrc.data(), oSrcSize, oDeviceSrc.pitch(), oSrcROI,
          oDeviceDst.data(), oDeviceDst.pitch(), oDstROI, aCoeffs, NPPI_INTER_NN));
  /**/


    // declare a host image for the result
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    // and copy the device result data into it
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    saveImage(sResultFilename, oHostDst);
    std::cout << "Saved image: " << sResultFilename << std::endl;

    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());

    exit(EXIT_SUCCESS);
  } catch (npp::Exception &rException) {
    std::cerr << "Program error! The following exception occurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
  } catch (...) {
    std::cerr << "Program error! An unknow type of exception occurred. \n";
    std::cerr << "Aborting." << std::endl;

    exit(EXIT_FAILURE);
    return -1;
  }

  return 0;
}
