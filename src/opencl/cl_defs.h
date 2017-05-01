#ifndef _DEFS_CL_H_
#define _DEFS_CL_H_

#include <iostream>

typedef enum {HOST_TO_DEVICE, DEVICE_TO_HOST, DEVICE_TO_DEVICE, HOST_TO_HOST} transfer_type;

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

__inline static const char* oclErrorString (cl_int error) {
	const int  errorcount = sizeof (errorstring) / sizeof (errorstring[0]);
	const int  index = -error;
	return (index >= 0 && index < errorcount) ? errorstring[index] : "";
}

#define STR_EXPAND(tok) #tok
#define STR(tok) STR_EXPAND(tok)

#define CL_SAFE_CALL(x)   do { cl_int ret = (x); clUtils::checkError(ret, STR(x)); } while (0)

#define DEFAULT_CL_DEVICE 0
#define MAX_BLOCKS 8192
#define BUFFER_STRING_LENGTH 256

#define MAX_SOURCE_SIZE (0x100000)

typedef struct {

	cl_device_id    id;
	cl_platform_id  platform;
	cl_device_type  type;
	std::string     device_string;
	std::string     platform_string;
	cl_uint         compute_units;
	cl_ulong        global_mem_size;
	cl_ulong        local_mem_size;
	cl_uint         preferred_vector;
	size_t 			profiling_timer_resolution;
	size_t          workitem_dims;
	size_t          workitem_size[3];
	size_t 			workgroup_size;

} cl_dev_info;

typedef struct {

	std::string     name;
	std::string     type;
	long : 32;      // padding
	int             typeId;
	std::string     subtype;
	long            localNum;
	int             globalNum;
	long : 32;      // padding

} deviceInfo;

/**
 * KernelWorkGroupInfo
 * class implements the functionality to query
 * various Kernel Work Group related parameters
 */

class KernelWorkGroupInfo {
  public:
	cl_ulong localMemoryUsed;           /**< localMemoryUsed amount of local memory used by kernel */
	size_t kernelWorkGroupSize;         /**< kernelWorkGroupSize Supported WorkGroup size as per OpenCL Runtime*/
	size_t compileWorkGroupSize[3];     /**< compileWorkGroupSize WorkGroup size as mentioned in kernel source */

	/**
	 * Constructor
	 */
	KernelWorkGroupInfo():
		localMemoryUsed(0),
		kernelWorkGroupSize(0) {
		compileWorkGroupSize[0] = 0;
		compileWorkGroupSize[1] = 0;
		compileWorkGroupSize[2] = 0;
	}

	/**
	 * setKernelWorkGroupInfo
	 * Set all information for a given device id
	 * @param kernel kernel object
	 * @param deviceId deviceID of the kernel object
	 * @return 0 if success else nonzero
	 */
	int setKernelWorkGroupInfo(cl_kernel& kernel, cl_device_id& deviceId) {
		cl_int status = CL_SUCCESS;
		//Get Kernel Work Group Info
		status = clGetKernelWorkGroupInfo(kernel,
		                                  deviceId,
		                                  CL_KERNEL_WORK_GROUP_SIZE,
		                                  sizeof(size_t),
		                                  &kernelWorkGroupSize,
		                                  NULL);
		if (checkVal(status, CL_SUCCESS,
		             "clGetKernelWorkGroupInfo failed(CL_KERNEL_WORK_GROUP_SIZE)")) {
			return 1;
		}
		status = clGetKernelWorkGroupInfo(kernel,
		                                  deviceId,
		                                  CL_KERNEL_LOCAL_MEM_SIZE,
		                                  sizeof(cl_ulong),
		                                  &localMemoryUsed,
		                                  NULL);
		if (checkVal(status, CL_SUCCESS,
		             "clGetKernelWorkGroupInfo failed(CL_KERNEL_LOCAL_MEM_SIZE)")) {
			return 1;
		}
		status = clGetKernelWorkGroupInfo(kernel,
		                                  deviceId,
		                                  CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
		                                  sizeof(size_t) * 3,
		                                  compileWorkGroupSize,
		                                  NULL);
		if (checkVal(status, CL_SUCCESS,
		             "clGetKernelWorkGroupInfo failed(CL_KERNEL_COMPILE_WORK_GROUP_SIZE)")) {
			return 1;
		}
		return 0;
	}
  private :

	/**
	 * checkVal
	 * Templated FunctionCheck whether any error occured
	 * @param input templated input
	 * @param reference templated input
	 * @param message string message
	 * @param isAPIerror bool optional variable
	 * @return 0 if success, else nonzero
	 */
	template<typename T>
	int checkVal(T input, T reference, std::string message,
	             bool isAPIerror = true) const {
		if (input == reference) {
			return 0;
		} else {
			if (isAPIerror) {
				std::cout << "Error: " << message << " Error code : ";
				std::cout << "getOpenCLErrorCodeStr(input)" << std::endl;
			} else {
				std::cout << message;
			}
			return 1;
		}
	}

};
#endif
