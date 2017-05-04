#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>

#include <opencl/cl_defs.h>
#include <utils.h>

#ifndef _CL_UTILS_
#define _CL_UTILS_

class clUtils {

  public:

	static std::vector <cl_dev_info> getAllDevices (cl_device_type dev_type = CL_DEVICE_TYPE_ALL);
	static std::vector <deviceInfo> listDevices (cl_device_type dev_type = CL_DEVICE_TYPE_ALL);
	static cl_dev_info getDevice (cl_device_id device);
	static void devInfo (cl_device_id device, int extended);
	static void checkError (const cl_int ciErrNum, const char* const operation);
	static const char* oclErrorString (cl_int error);

	// static int initResources(cl_resources* res, cl_device_id dev, unsigned int columns, unsigned inputLength, unsigned prox_synapses_per_column);
	// static void freeResources(cl_resources* res);

	static cl_program compileProgram (const char* const kernel_file, cl_context cxContext, cl_device_id d, const char*);
	static void runKernel (size_t localWorkSize, size_t globalWorkSize, cl_device_id device, cl_command_queue commandQueue, cl_kernel kernel, cl_event* gpuExecution);
	static void saveBinary (cl_program program, cl_device_id device, char* name);
};

std::vector <cl_dev_info> clUtils::getAllDevices (cl_device_type dev_type) {
	std::vector <cl_dev_info> available_devices;
	cl_device_id*
	devices;
	cl_uint
	ret_num_devices;
	cl_platform_id*
	platforms;
	cl_uint
	ret_num_platforms;
	// trace("Thread %d, Looking for OpenCL platforms...\n", threadID);
	// get the OpenCL platforms' number
	CL_SAFE_CALL (clGetPlatformIDs (0, NULL, &ret_num_platforms) );
	// trace("Thread %d, %d OpenCL platform(s) detected.\n", threadID,
	// ret_num_platforms);
	platforms = (cl_platform_id*) malloc (sizeof (cl_platform_id) * ret_num_platforms);

	if (!platforms) {
		// printf("Thread %d, Couldn't allocate memory for CL platforms", threadID);
		exit (-1);
	}

	// get the OpenCL platforms
	CL_SAFE_CALL (clGetPlatformIDs (ret_num_platforms, platforms, &ret_num_platforms) );
	// temp vars
	char name[BUFFER_STRING_LENGTH];
	char vendor[BUFFER_STRING_LENGTH];
	char version[BUFFER_STRING_LENGTH];
	char profile[BUFFER_STRING_LENGTH];

	for (unsigned i = 0; i < ret_num_platforms; i++) {
		CL_SAFE_CALL (clGetPlatformInfo (platforms[i], CL_PLATFORM_VENDOR, BUFFER_STRING_LENGTH, vendor, NULL) );
		CL_SAFE_CALL (clGetPlatformInfo (platforms[i], CL_PLATFORM_NAME, BUFFER_STRING_LENGTH, name, NULL) );
		CL_SAFE_CALL (clGetPlatformInfo (platforms[i], CL_PLATFORM_VERSION, BUFFER_STRING_LENGTH, version, NULL) );
		CL_SAFE_CALL (clGetPlatformInfo (platforms[i], CL_PLATFORM_PROFILE, BUFFER_STRING_LENGTH, profile, NULL) );
		clGetDeviceIDs (platforms[i], dev_type, 0, NULL, &ret_num_devices);

		if (ret_num_devices == 0) {
			std::cout << "No devices found supporting OpenCL.\n";
			exit (-1);

		} else {
			devices =
			    (cl_device_id*) malloc (sizeof (cl_device_id) * ret_num_devices);
			CL_SAFE_CALL (clGetDeviceIDs
			              (platforms[i], dev_type, ret_num_devices, devices,
			               &ret_num_devices) );

			for (unsigned j = 0; j < ret_num_devices; ++j) {
				cl_dev_info
				found_device;
				found_device = getDevice (devices[j]);
				found_device.id = devices[j];
				found_device.platform = platforms[i];
				std::string temp = std::string (name);
				found_device.platform_string = delUnnecessary (temp);
				available_devices.push_back (found_device);
			}

			free (devices);
		}
	}

	free (platforms);
	return available_devices;
}

std::vector <deviceInfo> clUtils::listDevices (cl_device_type dev_type) {
	std::vector <deviceInfo> devices;
	std::vector <cl_dev_info> available_devices = getAllDevices (dev_type);
	deviceInfo temp;

	for (unsigned i = 0; i < available_devices.size(); i++) {
		std::stringstream sstm;
		sstm << available_devices[i].device_string << " (" << available_devices[i].compute_units << " Compute Units, " << (float) available_devices[i].global_mem_size / (1024.0f * 1024.0f) << " MB)";
		temp.name = sstm.str();
		temp.localNum = (long) available_devices[i].id;
		temp.type = getFirstWord (available_devices[i].platform_string) + std::string (" OpenCL");
		temp.typeId = 0;
		devices.push_back (temp);
	}

	return devices;
}

cl_dev_info clUtils::getDevice (cl_device_id device) {

	cl_dev_info     found_device;
	char            device_string[BUFFER_STRING_LENGTH];
	cl_uint         compute_units;
	size_t          workgroup_size;
	cl_ulong        global_mem_size;
	cl_ulong        local_mem_size;
	cl_uint         preferred_vector;
	size_t			profiling_timer_resolution;

	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof (device_string), &device_string, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (compute_units), &compute_units, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (workgroup_size), &workgroup_size, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (global_mem_size), &global_mem_size, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof (local_mem_size), &local_mem_size, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof (cl_uint), &preferred_vector, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), &profiling_timer_resolution, NULL) );

	std::string temp = std::string (device_string);
	found_device.device_string = delUnnecessary (temp);
	found_device.compute_units = compute_units;
	found_device.workgroup_size = (unsigned) workgroup_size;
	found_device.global_mem_size = global_mem_size;
	found_device.local_mem_size = local_mem_size;
	found_device.preferred_vector = preferred_vector;
	found_device.profiling_timer_resolution = profiling_timer_resolution;

	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof (found_device.workitem_dims), &found_device.workitem_dims, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof (found_device.workitem_size), &found_device.workitem_size, NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (found_device.workgroup_size), &found_device.workgroup_size, NULL) );

	return found_device;
}

void clUtils::devInfo (cl_device_id device, int extended) {
	char            device_string[BUFFER_STRING_LENGTH];
	cl_bool         buf_bool;
	// CL_DEVICE_NAME
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_NAME, sizeof (device_string), &device_string, NULL) );
	printf ("\t\t\tCL_DEVICE_NAME = %s\n", device_string);
	// CL_DEVICE_VENDOR
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_VENDOR, sizeof (device_string), &device_string, NULL) );
	printf ("\t\t\tCL_DEVICE_VENDOR = %s\n", device_string);
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_VERSION, sizeof (device_string), &device_string, NULL) );
	printf ("\t\t\tCL_DEVICE_VERSION = %s\n", device_string);
	// CL_DRIVER_VERSION
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DRIVER_VERSION, sizeof (device_string), &device_string, NULL) );
	printf ("\t\t\tCL_DRIVER_VERSION = %s\n", device_string);
	// CL_DEVICE_INFO
	cl_device_type  type;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_TYPE, sizeof (type), &type, NULL) );

	if (type & CL_DEVICE_TYPE_CPU) printf ("\t\t\tCL_DEVICE_TYPE = CL_DEVICE_TYPE_CPU\n");
	if (type & CL_DEVICE_TYPE_GPU) printf ("\t\t\tCL_DEVICE_TYPE = CL_DEVICE_TYPE_GPU\n");
	if (type & CL_DEVICE_TYPE_ACCELERATOR) printf ("\t\t\tCL_DEVICE_TYPE = CL_DEVICE_TYPE_ACCELERATOR\n");
	if (type & CL_DEVICE_TYPE_DEFAULT) printf ("\t\t\tCL_DEVICE_TYPE = CL_DEVICE_TYPE_DEFAULT\n");

	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_AVAILABLE, sizeof (buf_bool), &buf_bool, NULL) );

	if (extended) { printf ("\t\t\tCL_DEVICE_AVAILABLE = %s\n", buf_bool == CL_TRUE ? "YES" : "NO"); }

	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_COMPILER_AVAILABLE, sizeof (buf_bool), &buf_bool, NULL) );

	if (extended) { printf ("\t\t\tCL_DEVICE_COMPILER_AVAILABLE = %s\n", buf_bool == CL_TRUE ? "YES" : "NO"); }

	// CL_DEVICE_MAX_COMPUTE_UNITS
	cl_uint         compute_units;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof (compute_units), &compute_units, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_COMPUTE_UNITS = %u\n", compute_units);
	// CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
	size_t          workitem_dims;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof (workitem_dims), &workitem_dims, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = %u\n", (unsigned) workitem_dims);
	// CL_DEVICE_MAX_WORK_ITEM_SIZES
	size_t          workitem_size[3];
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof (workitem_size), &workitem_size, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_WORK_ITEM_SIZES = %u / %u / %u \n", (unsigned) workitem_size[0], (unsigned) workitem_size[1], (unsigned) workitem_size[2]);
	// CL_DEVICE_MAX_WORK_GROUP_SIZE
	size_t          workgroup_size;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof (workgroup_size), &workgroup_size, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_WORK_GROUP_SIZE = %u\n", (unsigned) workgroup_size);
	// CL_DEVICE_MAX_CLOCK_FREQUENCY
	cl_uint         clock_frequency;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof (clock_frequency), &clock_frequency, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_CLOCK_FREQUENCY = %u MHz\n", clock_frequency);
	// CL_DEVICE_ADDRESS_BITS
	cl_uint         addr_bits;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_ADDRESS_BITS, sizeof (addr_bits), &addr_bits, NULL) );

	if (extended) printf ("\t\t\tCL_DEVICE_ADDRESS_BITS = %u\n", addr_bits);

	// CL_DEVICE_MAX_MEM_ALLOC_SIZE
	cl_ulong        max_mem_alloc_size;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof (max_mem_alloc_size), &max_mem_alloc_size, NULL) );

	if (extended) { printf ("\t\t\tCL_DEVICE_MAX_MEM_ALLOC_SIZE = %u MB\n", (unsigned int) (max_mem_alloc_size / (1024 * 1024) ) ); }

	// CL_DEVICE_GLOBAL_MEM_SIZE
	cl_ulong        mem_size;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof (mem_size), &mem_size, NULL) );
	printf ("\t\t\tCL_DEVICE_GLOBAL_MEM_SIZE = %u MB\n",
	        (unsigned int) (mem_size / (1024 * 1024) ) );
	// TODO
	// CL_DEVICE_GLOBAL_MEM_CACHE_SIZE
	// CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
	// CL_DEVICE_ERROR_CORRECTION_SUPPORT
	cl_bool         error_correction_support;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof (error_correction_support), &error_correction_support, NULL) );
	printf ("\t\t\tCL_DEVICE_ERROR_CORRECTION_SUPPORT = %s\n", error_correction_support == CL_TRUE ? "YES" : "NO");
	// CL_DEVICE_LOCAL_MEM_TYPE
	cl_device_local_mem_type local_mem_type;
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof (local_mem_type), &local_mem_type, NULL) );

	if (extended) { printf ("\t\t\tCL_DEVICE_LOCAL_MEM_TYPE = %s\n", local_mem_type == 1 ? "LOCAL" : "GLOBAL"); }

	// CL_DEVICE_LOCAL_MEM_SIZE
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof (mem_size), &mem_size, NULL) );
	printf ("\t\t\tCL_DEVICE_LOCAL_MEM_SIZE = %u kB\n", (unsigned int) (mem_size / 1024) );
	// CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof (mem_size), &mem_size, NULL) );
	printf ("\t\t\tCL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = %u kB\n", (unsigned int) (mem_size / 1024) );

	// CL_DEVICE_QUEUE_PROPERTIES
	cl_command_queue_properties queue_properties;
	CL_SAFE_CALL (clGetDeviceInfo
	              (device, CL_DEVICE_QUEUE_PROPERTIES, sizeof (queue_properties),
	               &queue_properties, NULL) );

	if (extended)
		if (queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
			printf ("\t\t\tCL_DEVICE_QUEUE_PROPERTIES = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE\n");
		}

	if (extended) {
		if (queue_properties & CL_QUEUE_PROFILING_ENABLE) printf ("\t\t\tCL_DEVICE_QUEUE_PROPERTIES = CL_QUEUE_PROFILING_ENABLE\n");

		// CL_DEVICE_IMAGE_SUPPORT
		cl_bool         image_support;
		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_IMAGE_SUPPORT, sizeof (image_support), &image_support, NULL) );

		if (extended) printf ("\t\t\tCL_DEVICE_IMAGE_SUPPORT = %u\n", image_support);

		// CL_DEVICE_MAX_READ_IMAGE_ARGS
		cl_uint         max_read_image_args;
		CL_SAFE_CALL (clGetDeviceInfo(device, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof (max_read_image_args), &max_read_image_args, NULL) );

		if (extended) printf ("\t\t\tCL_DEVICE_MAX_READ_IMAGE_ARG = %u\n", max_read_image_args);

		// CL_DEVICE_MAX_WRITE_IMAGE_ARGS
		cl_uint         max_write_image_args;
		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof (max_write_image_args), &max_write_image_args, NULL) );

		if (extended) printf ("\t\t\tCL_DEVICE_MAX_WRITE_IMAGE_ARGS = %u\n", max_write_image_args);

		// CL_DEVICE_IMAGE2D_MAX_WIDTH, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
		// CL_DEVICE_IMAGE3D_MAX_WIDTH, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
		// CL_DEVICE_IMAGE3D_MAX_DEPTH
		size_t szMaxDims[5];

		if (extended) printf ("\t\t\tCL_DEVICE_IMAGE:\n");
		CL_SAFE_CALL (clGetDeviceInfo
		              (device, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof (size_t), &szMaxDims[0],
		               NULL) );

		if (extended) printf ("\t\t\t\t2D_MAX_WIDTH = %u\n", (unsigned) szMaxDims[0]);

		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof (size_t), &szMaxDims[1], NULL) );
		if (extended) printf ("\t\t\t\t2D_MAX_HEIGHT = %u\n", (unsigned) szMaxDims[1]);

		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof (size_t), &szMaxDims[2], NULL) );
		if (extended) printf ("\t\t\t\t3D_MAX_WIDTH = %u\n", (unsigned) szMaxDims[2]);

		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof (size_t), &szMaxDims[3], NULL) );
		if (extended) printf ("\t\t\t\t3D_MAX_HEIGHT = %u\n", (unsigned) szMaxDims[3]);

		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof (size_t), &szMaxDims[4], NULL) );
		if (extended) printf ("\t\t\t\t3D_MAX_DEPTH = %u\n", (unsigned) szMaxDims[4]);

		// CL_DEVICE_EXTENSIONS: get device extensions, and if any then parse & log the
		// string onto separate lines
		CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_EXTENSIONS, sizeof (device_string), &device_string, NULL) );

		if (strlen (device_string) != 0) {
			char           delimiter[] = " ";
			unsigned long  inputLength = strlen (device_string);
			char*           word,
			                *context;
			char*           inputCopy = (char*) calloc (inputLength + 1, sizeof (char) );
			strncpy (inputCopy, device_string, inputLength);

			if (extended) printf ("\t\t\tCL_DEVICE_EXTENSIONS:\n");

			word = strtok_r (inputCopy, delimiter, &context);

			if (extended) printf ("\t\t\t\t%s\n", word);

			while (1) {
				word = strtok_r (NULL, delimiter, &context);

				if (word == NULL)
					break;

				if (extended)
					printf ("\t\t\t\t%s\n", word);
			}

			free (inputCopy);
		}
	}

	// CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
	if (extended)
		printf ("\t\t\tCL_DEVICE_PREFERRED_VECTOR_WIDTH:\t\n");

	cl_uint         vec_width[6];
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof (cl_uint), &vec_width[0], NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof (cl_uint), &vec_width[1], NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof (cl_uint), &vec_width[2], NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof (cl_uint), &vec_width[3], NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof (cl_uint), &vec_width[4], NULL) );
	CL_SAFE_CALL (clGetDeviceInfo (device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof (cl_uint), &vec_width[5], NULL) );

	if (extended) printf ("\t\t\t\tCHAR = %u\n\t\t\t\tSHORT = %u\n\t\t\t\tINT = %u\n\t\t\t\tFLOAT = %u\n\t\t\t\tDOUBLE = %u\n", vec_width[0], vec_width[1], vec_width[2], vec_width[3], vec_width[4]);

}

const char*
clUtils::oclErrorString (cl_int error) {
	static const char* errorstring[] = {
		"cl_success",
		"cl_device_not_found",
		"cl_device_not_available",
		"cl_compiler_not_available",
		"cl_mem_object_allocation_failure",
		"cl_out_of_resources",
		"cl_out_of_host_memory",
		"cl_profiling_info_not_available",
		"cl_mem_copy_overlap",
		"cl_image_format_mismatch",
		"cl_image_format_not_supported",
		"cl_build_program_failure",
		"cl_map_failure",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"cl_invalid_value",
		"cl_invalid_device_type",
		"cl_invalid_platform",
		"cl_invalid_device",
		"cl_invalid_context",
		"cl_invalid_queue_properties",
		"cl_invalid_command_queue",
		"cl_invalid_host_ptr",
		"cl_invalid_mem_object",
		"cl_invalid_image_format_descriptor",
		"cl_invalid_image_size",
		"cl_invalid_sampler",
		"cl_invalid_binary",
		"cl_invalid_build_options",
		"cl_invalid_program",
		"cl_invalid_program_executable",
		"cl_invalid_kernel_name",
		"cl_invalid_kernel_definition",
		"cl_invalid_kernel",
		"cl_invalid_arg_index",
		"cl_invalid_arg_value",
		"cl_invalid_arg_size",
		"cl_invalid_kernel_args",
		"cl_invalid_work_dimension",
		"cl_invalid_work_group_size",
		"cl_invalid_work_item_size",
		"cl_invalid_global_offset",
		"cl_invalid_event_wait_list",
		"cl_invalid_event",
		"cl_invalid_operation",
		"cl_invalid_gl_object",
		"cl_invalid_buffer_size",
		"cl_invalid_mip_level",
		"cl_invalid_global_work_size",
	};
	const int       errorcount = sizeof (errorstring) / sizeof (errorstring[0]);
	const int       index = -error;
	return (index >= 0 && index < errorcount) ? errorstring[index] : "";
}

void clUtils::checkError (const cl_int ciErrNum, const char* const operation) {
	if (ciErrNum != CL_SUCCESS) {
		printf ("ERROR: %s failed, %s, %d\n", operation, oclErrorString (ciErrNum), ciErrNum );
		// exit(EXIT_FAILURE);
	}
}

void check_error (cl_int errcode, const char* msg, ...) {
	if (errcode < 0) {
		char formatted[1024];

		if (msg == NULL) {
			fprintf (stderr, "Error %d: %s\n", errcode, oclErrorString (errcode) );

		} else {
			va_list args;
			va_start (args, msg);
			vsprintf (formatted, msg, args);
			va_end (args);
			fprintf (stderr, "Error %d: %s\n", errcode, formatted);
		}

		exit (EXIT_FAILURE);
	}
}

cl_program clUtils::compileProgram (const char* const kernel_file, cl_context cxContext, cl_device_id d, const char* flags = "") {
	cl_program cpProgram = NULL;
	size_t program_length = 0;
	char* const source = readFile (kernel_file, &program_length);
	cl_int ciErrNum;

	if (source) {
		// Create the program for all GPUs in the context
		cpProgram = clCreateProgramWithSource (cxContext, 1, (const char**) &source, &program_length, &ciErrNum);
		free (source);
		clUtils::checkError (ciErrNum, "clCreateProgramWithSource");
		/*
		    Build program
		*/
		char clcompileflags[1024];
		sprintf (clcompileflags, "-cl-fast-relaxed-math -cl-mad-enable %s", flags);
		//sprintf ( clcompileflags, "-cl-mad-enable" );
		ciErrNum = clBuildProgram (cpProgram, 0, NULL, clcompileflags, NULL, NULL);
		clUtils::checkError (ciErrNum, "clBuildProgram");

		if (ciErrNum != CL_SUCCESS) {
			char*           build_log;
			size_t          log_size;
			// First call to know the proper size
			ciErrNum = clGetProgramBuildInfo (cpProgram, d, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
			clUtils::checkError (ciErrNum, "clGetProgramBuildInfo 1");
			build_log = (char*) malloc ( (log_size + 1) );
			// Second call to get the log
			ciErrNum = clGetProgramBuildInfo (cpProgram, d, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
			clUtils::checkError (ciErrNum, "clGetProgramBuildInfo 2");
			build_log[log_size] = '\0';
			printf ("--- Build log extended kernel---\n ");
			printf ("%s\n", build_log);
			free (build_log);
			exit (1);
		}
	}

	return cpProgram;
}

#endif
