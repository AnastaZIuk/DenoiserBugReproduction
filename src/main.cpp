#include "config/BuildConfigOptions.h"
#include "nvidia/CheckMacros.h"

#include <iostream>
#include <vector>
#include <string>
#include <string_view>
#include <filesystem>
#include <array>
#include <map>
#include <cassert>

#include <cuda.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include "gli/gli.hpp"

void CU_CHECK(const CUresult result)
{
	if (result != CUDA_SUCCESS)
	{
		const char* name;
		cuGetErrorName(result, &name);
		std::cerr << "ERROR: Failed with " << name << " (" << result << ")\n";
		MY_ASSERT(!"CU_CHECK fatal");
	}
}

void OPTIX_CHECK(const OptixResult result)
{
	if (result != OPTIX_SUCCESS)
	{
		std::cerr << "ERROR: Failed with (" << result << ")\n";
		MY_ASSERT(!"OPTIX_CHECK fatal");
	}
}

constexpr uint32_t overlap = 64;
//constexpr uint32_t tileWidth = 1920/2, tileHeight = 1080/2;
constexpr uint32_t tileWidth = 1024, tileHeight = 1024;
constexpr uint32_t tileWidthWithOverlap = tileWidth + overlap * 2;
constexpr uint32_t tileHeightWithOverlap = tileHeight + overlap * 2;
constexpr uint32_t outputDimensions[] = { tileWidth , tileHeight };

void DBROptixDefaultCallback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
	uint32_t contextID = reinterpret_cast<const uint32_t&>(cbdata);
	printf("OptiX Context:%d [%s]: %s\n", contextID, tag, message);
}


int main()
{
	CU_CHECK(cuInit(0));
	OPTIX_CHECK(optixInit());

	bool status = true;

	constexpr std::array<std::string_view, 3> hardcodedInputs =
	{
		"spp_benchmark_4k_512_reference_optix_input_albedo.dds",
		"spp_benchmark_4k_512_reference_optix_input_color.dds",
		"spp_benchmark_4k_512_reference_optix_input_normal.dds"
	};

	uint32_t maxResolution[2] = { 0,0 };
	std::array<gli::texture, hardcodedInputs.size()> inputKindTextures;
	gli::texture outputTexture;
	{
		uint8_t offset = {};
		for (auto& hardcodedInput : hardcodedInputs)
		{
			const std::string inputFile = DBR_ROOT + std::string("/data/") + hardcodedInput.data();

			inputKindTextures[offset] = gli::load_dds(inputFile);
			status = !inputKindTextures[offset].empty();
			assert(status); // Input hasn't been loaded!

			auto extent = inputKindTextures[offset].extent(0);

			maxResolution[0] = std::max<uint32_t>(extent.x, maxResolution[0]);
			maxResolution[1] = std::max<uint32_t>(extent.y, maxResolution[1]);

			if (offset == 0u)
				outputTexture = inputKindTextures[offset]; // For copying header data

			++offset;
		}
	}

	CUdevice device;
	CU_CHECK(cuDeviceGet(&device,0u)); // it won't work

	// create context
	CUcontext context;
	CU_CHECK(cuCtxCreate_v2(&context,CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX,device));
	{
		uint32_t version = 0u;

		CU_CHECK(cuCtxGetApiVersion(context, &version));

		if (version < 3020)
		{
			status = false;
			assert(status);
			return 1;
		}

		CU_CHECK(cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_L1));
	}

	CUstream stream;
	CU_CHECK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));

	/*
		Init Optix Context
	*/

	OptixDeviceContext optixContext;
	OPTIX_CHECK(optixDeviceContextCreate(context, {}, &optixContext));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(optixContext, DBROptixDefaultCallback, reinterpret_cast<void*>(context), 3));

	/*
		Creating Denoisers
	*/

	constexpr auto forcedOptiXFormat = OPTIX_PIXEL_FORMAT_HALF3;
	constexpr auto forcedOptiXFormatPixelStride = 6u;

	auto createDenoiser = [&](const OptixDenoiserOptions* options, OptixDenoiserModelKind model = OPTIX_DENOISER_MODEL_KIND_HDR, void* modelData = nullptr, size_t modelDataSizeInBytes = 0ull) -> OptixDenoiser
	{
		if (!options)
			return nullptr;

		OptixDenoiser denoiser = nullptr;
		OPTIX_CHECK(optixDenoiserCreate(optixContext, options, &denoiser));

		if (!denoiser)
			return nullptr;

		if(optixDenoiserSetModel(denoiser, model, modelData, modelDataSizeInBytes) != OPTIX_SUCCESS)
			return nullptr;

		return denoiser;
	};

	constexpr uint32_t kInputBufferCount = 3u;
	OptixDenoiserOptions opts = { OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL };
	OptixDenoiser denoiser = createDenoiser(&opts);
	if (!denoiser)
		status = false;
	assert(status); // Could not create Optix Color-Albedo-Normal Denoiser!

	/*
		Compute memory resources for denoiser
	*/

	size_t denoiserStateBufferSize = 0ull;
	size_t scratchBufferSize = 0ull;
	size_t singleInputBufferSize = 0ull;
	{
		OptixDenoiserSizes denoiserMemReqs;

		OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser, outputDimensions[0], outputDimensions[1], &denoiserMemReqs));

		denoiserStateBufferSize = denoiserMemReqs.stateSizeInBytes;
		scratchBufferSize = denoiserMemReqs.withOverlapScratchSizeInBytes;
		singleInputBufferSize = forcedOptiXFormatPixelStride * maxResolution[0] * maxResolution[1];
	}
	const size_t pixelBufferSize = singleInputBufferSize*kInputBufferCount;

	std::string message = "Total VRAM consumption for Denoiser algorithm: ";
	std::cout << message + std::to_string(denoiserStateBufferSize + scratchBufferSize + pixelBufferSize);

	if (pixelBufferSize == 0u)
		status = false;

	assert(status); // No input files at all!

	CUdeviceptr denoiserState;

	CU_CHECK(cuMemAlloc(&denoiserState, denoiserStateBufferSize));

	CUdeviceptr	imageIntensity;
	CU_CHECK(cuMemAlloc(&imageIntensity, sizeof(float)));

	CUdeviceptr inputPixelBuffer;
	CU_CHECK(cuMemAlloc(&inputPixelBuffer, pixelBufferSize));
	CUdeviceptr inputPixelBuffers[3] =
	{
		inputPixelBuffer,
		inputPixelBuffer+singleInputBufferSize,
		inputPixelBuffer+singleInputBufferSize*2u
	};

	/*
		Fill CUDA buffers with appropriate texture data
	*/

	{
		uint8_t offset = {}; // TODO: because you're declaring to OptiX that data is RGB not RGBA (HALF3) you either need to map the memory of the allocated buffers and write the copy loop yourself, or do it yourself/GLI on the CPU before memcpy to GPU
		for (auto& inputKindTexture : inputKindTextures)
		{
			void* ptrToBegginingOfData = inputKindTexture.data(0, 0, 0);
			CU_CHECK(cuMemcpyHtoD_v2(inputPixelBuffers[offset++], ptrToBegginingOfData, singleInputBufferSize));
			inputKindTexture.clear();
		}
	}

	// TODO: denoise

	status = gli::save_dds(outputTexture, "outputResult.dds");
	assert(status); // Could not save output texture!
}