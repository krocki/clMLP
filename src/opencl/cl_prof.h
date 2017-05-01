/*
    @Author: kmrocki@us.ibm.com
    @Date:   2017-04-28 12:49:03
    @Last Modified by:   kmrocki@us.ibm.com
    @Last Modified time: 2017-04-29 11:21:31
*/

#ifndef __CL_PROF_H__
#define __CL_PROF_H__

#include <string>
#include <utils.h>
#include <containers/dict.h>

typedef enum profiling_type {OFF = 0, CPU_ONLY = 1, GPU_ONLY = 2, CPU_GPU = 3} profiling_type;

profiling_type prof_enabled;

#define CL_PROF_ENABLED (((prof_enabled == GPU_ONLY) || (prof_enabled == CPU_GPU)) ? true : false)
#define CPU_PROF_ENABLED (((prof_enabled == CPU_ONLY) || (prof_enabled == CPU_GPU)) ? true : false)
#define GPU_BLOCKS_CPU 0

cl_command_queue defqueue;

// using this for profiling
#define _TIMED_CALL_(func)  \
	do { \
		std::chrono::time_point<std::chrono::system_clock> func_start, func_end; \
		if (GPU_BLOCKS_CPU) clFinish (defqueue); \
		if (CPU_PROF_ENABLED) { \
			func_start = std::chrono::system_clock::now(); \
		} \
		func; \
		if (GPU_BLOCKS_CPU) clFinish (defqueue); \
		if (CPU_PROF_ENABLED) { \
			func_end = std::chrono::system_clock::now(); \
			double func_time = ( double ) std::chrono::duration_cast<std::chrono::nanoseconds> ( func_end - func_start ).count(); \
			pdata[#func].time += func_time; \
		} \
	} while (0)

typedef enum sort_method_type { NO_SORTING = 0, SORT_BY_TIME_DESC = 1, SORT_BY_FLOPS_DESC = 2, SORT_BY_NAME = 3, SORT_BY_NAME_DESC = 4} sort_method_type;
std::chrono::time_point<std::chrono::system_clock> start, end;


class prof_data {

  public:

	std::string key = "";
	std::string description = "";

	long double time = 0.0;
	long double flops = 0.0;
	long double bytes_in = 0.0;
	long double bytes_out = 0.0;

	prof_data (std::string _description = "") : description (_description) {
		reset();
	}

	void reset() {
		time = 0.0;
		flops = 0.0;
		bytes_in = 0.0;
		bytes_out = 0.0;
	}

	void show (const double global_time = 0.0, double total_cl_time = 0.0, unsigned long total_cl_flops_performed = 0L, unsigned long total_bytes_in = 0L,  unsigned long total_bytes_out = 0L, const int m = 7, const int n = 3) {

		double cl_time_perc = 0;
		double total_time_perc = 0;
		double total_flops_perc = 0;
		double total_bytes_in_perc = 0;
		double total_bytes_out_perc = 0;

		if (!description.empty() ) std::cout << ", descr: " << description;
		if (total_cl_time > 0.0) cl_time_perc = (100.0 * time) / total_cl_time;
		if (global_time > 0.0) total_time_perc = (1e-7 * time) / global_time;
		if (total_cl_flops_performed > 0) total_flops_perc = ( (100.0 * (long double) flops) / (long double) total_cl_flops_performed);
		if (total_bytes_in > 0) total_bytes_in_perc = ( (100.0 * (long double) bytes_in) / (long double) total_bytes_in);
		if (total_bytes_out > 0) total_bytes_out_perc = ( (100.0 * (long double) bytes_out) / (long double) total_bytes_out);

		std::cout << ", time: " << to_string_with_precision (time * 1e-9, m, n) << " s ";
		std::cout << " / ( " << to_string_with_precision (cl_time_perc, 6, 2) << "% / ";
		std::cout << to_string_with_precision (total_time_perc, 6, 2) << "% )";
		std::cout << ", GFlOPs " << to_string_with_precision (flops * 1e-9, m, n) << ", ";
		std::cout << " ( " << to_string_with_precision (total_flops_perc, 6, 2) << "% )";
		std::cout << ", GF/s: " << to_string_with_precision (flops / time, m, n) << "/s ";
		std::cout << ", GB in " << to_string_with_precision (bytes_in * 1e-9, m, n) << ", ";
		std::cout << "( " << to_string_with_precision (total_bytes_in_perc, 6, 2) << "% )";
		std::cout << ", GB/s: " << to_string_with_precision (bytes_in / time, m, n) << " ";
		std::cout << ", GB out " << to_string_with_precision (bytes_out * 1e-9, m, n) << ", ";
		std::cout << " ( " << to_string_with_precision (total_bytes_out_perc, 6, 2) << "% )";
		std::cout << ", GB/s: " << to_string_with_precision (bytes_out / time, m, n) << " ";
		std::cout << std::endl;
	}
};

void show_profiling_data (Dict<prof_data>& pdata, sort_method_type sort_method = SORT_BY_TIME_DESC, profiling_type ptype = OFF) {

	unsigned long total_cl_flops_performed  = 0L;
	unsigned long total_bytes_in  = 0L;
	unsigned long total_bytes_out  = 0L;
	double total_cl_time = 0.0;
	end = std::chrono::system_clock::now();
	double difference = (double) std::chrono::duration_cast<std::chrono::microseconds> (end - start).count() / (double) 1e6;
	std::cout << "T = " << difference << " s" << std::endl;

	//first pass
	for (size_t i = 0; i < pdata.entries.size(); i ++) {
		if (ptype != OFF) {
			total_cl_flops_performed += pdata.entries[ i ].flops;
			total_cl_time += pdata.entries[ i ].time;
			total_bytes_in += pdata.entries[ i ].bytes_in;
			total_bytes_out += pdata.entries[ i ].bytes_out;
		}
	}

	std::vector<size_t> sorted_idxs;

	switch (sort_method) {
	case NO_SORTING:
		sorted_idxs.resize (pdata.entries.size() );
		std::iota (sorted_idxs.begin(), sorted_idxs.end(), 0);
		break;

	case SORT_BY_TIME_DESC:
		sorted_idxs = pdata.sorted_idxs ([&] (size_t i1, size_t i2) {
			return pdata.entries[i1].time > pdata.entries[i2].time;
		});
		break;

	case SORT_BY_FLOPS_DESC:
		sorted_idxs = pdata.sorted_idxs ([&] (size_t i1, size_t i2) {
			return pdata.entries[i1].flops / pdata.entries[i1].time > pdata.entries[i2].flops / pdata.entries[i2].time;
		});
		break;

	case SORT_BY_NAME:
		sorted_idxs = pdata.sorted_idxs ([&] (size_t i1, size_t i2) {
			return pdata.entries[i1].key < pdata.entries[i2].key;
		});
		break;

	case SORT_BY_NAME_DESC:
		sorted_idxs = pdata.sorted_idxs ([&] (size_t i1, size_t i2) {
			return pdata.entries[i1].key > pdata.entries[i2].key;
		});
		break;
	}

	//second pass
	for (size_t i = 0; i < pdata.entries.size(); i ++) {
		if (ptype != OFF) {
			std::cout << std::setw (50) << pdata.reverse_namemap[ sorted_idxs[i] ];
			pdata.entries[ sorted_idxs[i] ].show (difference, total_cl_time, total_cl_flops_performed, total_bytes_in, total_bytes_out);
		}

		pdata.entries[ sorted_idxs[i] ].reset();
	}

	if (ptype != OFF) {
		std::cout << std::endl;
		std::cout << "Total profiled time: " << 1e-9 * total_cl_time << " s " << "( " << to_string_with_precision ( (100.0 * (long double) (1e-9 * total_cl_time) / (long double) difference), 7, 3) << "% ) " << std::endl;
		std::cout << "Total compute: " << 1e-9 * ( (long double) total_cl_flops_performed / (long double) difference) << " GF/s" << std::endl;
		std::cout << "Total in: " << 1e-9 * ( (long double) total_bytes_in) << " GB" << std::endl;
		std::cout << "Total out: " << 1e-9 * ( (long double) total_bytes_out) << " GB" << std::endl;
		std::cout << std::endl;
	}

	start = std::chrono::system_clock::now();
}

Dict<prof_data> pdata;

#endif /*__CL_PROF_H__*/
