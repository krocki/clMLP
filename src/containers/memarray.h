/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-05-01 22:23:52
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-05-02 20:46:13
*/

#include <cstddef>

#ifndef __MEMARRAY__
#define __MEMARRAY__

// !TODO!
//abstract class for storing arrays
template <typename T = float>
class memarray {

  public:

	virtual size_t rows() const = 0;
	virtual size_t cols() const = 0;
	virtual size_t length() const = 0;
	virtual void resize (size_t rows, size_t cols) = 0;

	virtual void sync_device() {};
	virtual void sync_host() {};

	virtual void free_device_mem() = 0;

};

#endif