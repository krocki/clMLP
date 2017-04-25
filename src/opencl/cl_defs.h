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
#define CL_SAFE_CALL(x)   do { \
        cl_int ret = (x); clUtils::checkError(ret, STR(x)); \
    } while (0)

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
    unsigned        workgroup_size;
    cl_ulong        global_mem_size;
    cl_ulong        local_mem_size;
    cl_uint         preferred_vector;
    long : 32;      // padding

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

#endif
