/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-28 12:49:03
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-28 13:31:47
*/

#ifndef __CL_PROF_H__
#define __CL_PROF_H__

#include <string>
#include <utils.h>

class prof_data {

  public:

	std::string description = "";
	long double time = 0.0;
	long double flops = 0.0;
	long double bytes_in = 0.0;
	long double bytes_out = 0.0;

	prof_data(std::string _description = "") : description(_description) { reset(); }

	void reset() {

		time = 0.0;
		flops = 0.0;
		bytes_in = 0.0;
		bytes_out = 0.0;

	}

	void show(const int m = 8, const int n = 2) {

		if (!description.empty()) std::cout << ", descr: " << description;

		std::cout << ", total time: " << to_string_with_precision (time * 1e-9, m, n) << " s";
		std::cout << ", GFlOPs " << to_string_with_precision (flops * 1e-9, m, n) << ", GFlOP/s " << to_string_with_precision (flops / time, m, n);
		std::cout << ", GB in " << to_string_with_precision (bytes_in * 1e-9, m, n) << ", GB in/s " << to_string_with_precision (bytes_in / time, m, n);
		std::cout << ", GB out " << to_string_with_precision (bytes_out * 1e-9, m, n) << ", GB in/s " << to_string_with_precision (bytes_out / time, m, n);

		std::cout << std::endl;

	}
};

#endif /*__CL_PROF_H__*/