/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-28 12:49:03
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-28 16:32:20
*/

#ifndef __CL_PROF_H__
#define __CL_PROF_H__

#include <string>
#include <utils.h>

typedef enum sort_method_type { NO_SORTING = 0, SORT_BY_TIME_DESC = 1, SORT_BY_FLOPS_DESC = 2, SORT_BY_NAME = 3, SORT_BY_NAME_DESC = 4} sort_method_type;

class prof_data {

	public:
	
		std::string key = "";
		std::string description = "";
		
		long double time = 0.0;
		long double flops = 0.0;
		long double bytes_in = 0.0;
		long double bytes_out = 0.0;
		
		prof_data ( std::string _description = "" ) : description ( _description ) { reset(); }
		
		void reset() {
		
			time = 0.0;
			flops = 0.0;
			bytes_in = 0.0;
			bytes_out = 0.0;
			
		}
		
		void show ( const double global_time = 0.0, double total_cl_time = 0.0, unsigned long total_cl_flops_performed = 0L, unsigned long total_bytes_in = 0L,  unsigned long total_bytes_out = 0L,
					const int m = 7, const int n = 3 ) {
					
			double cl_time_perc = 0;
			double total_time_perc = 0;
			double total_flops_perc = 0;
			double total_bytes_in_perc = 0;
			double total_bytes_out_perc = 0;
			
			if ( !description.empty() ) std::cout << ", descr: " << description;
			if ( total_cl_time > 0.0 ) cl_time_perc = ( 100.0 * time ) / total_cl_time;
			if ( global_time > 0.0 ) total_time_perc = ( 1e-7 * time ) / global_time;
			if ( total_cl_flops_performed > 0 ) total_flops_perc = ( ( 100.0 * ( long double ) flops ) / ( long double ) total_cl_flops_performed );
			if ( total_bytes_in > 0 ) total_bytes_in_perc = ( ( 100.0 * ( long double ) bytes_in ) / ( long double ) total_bytes_in );
			if ( total_bytes_out > 0 ) total_bytes_out_perc = ( ( 100.0 * ( long double ) bytes_out ) / ( long double ) total_bytes_out );
			
			std::cout << ", time: " << to_string_with_precision ( time * 1e-9, m, n ) << " s ";
			std::cout << " / ( " << to_string_with_precision ( cl_time_perc, 6, 2 ) << "% / ";
			std::cout << to_string_with_precision ( total_time_perc, 6, 2 ) << "% )";
			
			std::cout << ", GFlOPs " << to_string_with_precision ( flops * 1e-9, m, n ) << ", ";
			std::cout << " ( " << to_string_with_precision ( total_flops_perc, 6, 2 ) << "% )";
			std::cout << ", GF/s: " << to_string_with_precision ( flops / time, m, n ) << "/s ";
			
			std::cout << ", GB in " << to_string_with_precision ( bytes_in * 1e-9, m, n ) << ", ";
			std::cout << "( " << to_string_with_precision ( total_bytes_in_perc, 6, 2 ) << "% )";
			std::cout << ", GB/s: " << to_string_with_precision ( bytes_in / time, m, n ) << " ";
			
			std::cout << ", GB out " << to_string_with_precision ( bytes_out * 1e-9, m, n ) << ", ";
			std::cout << " ( " << to_string_with_precision ( total_bytes_out_perc, 6, 2 ) << "% )";
			std::cout << ", GB/s: " << to_string_with_precision ( bytes_out / time, m, n ) << " ";
			
			std::cout << std::endl;
			
		}
};

#endif /*__CL_PROF_H__*/
