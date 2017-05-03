/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-05-01 22:22:38
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-05-02 20:46:23
*/

#include <containers/memarray.h>
#include <utils.h>
#include <Eigen/Dense>

#ifndef __EIGEN_MATRIX__
#define __EIGEN_MATRIX__

// !TODO!

template<class T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T = float>
class eigen_matrix : public memarray<T> {

	host_matrix<T> host_data;
	host_matrix<T>& ref_host_data;

	eigen_matrix (Eigen::Vector2i size) {
		host_data = host_matrix<T> (size[0], size[1]);
		host_data.setZero();
		ref_host_data = host_data;

	};

	eigen_matrix (host_matrix<T>& m) {
		host_data = m;
		ref_host_data = host_data;
		alloc_device_mem();
	};

	eigen_matrix& operator= (const eigen_matrix& other) {
		host_data = other.host_data;
		ref_host_data = host_data;
		return *this;
	};

	void resize (size_t rows, size_t cols) { host_data.resize (rows, cols); }

	T sum() { return ref_host_data.sum(); }

	size_t rows() const { return ref_host_data.rows(); }
	size_t cols() const { return ref_host_data.cols(); }
	size_t length() const { return ref_host_data.rows() * ref_host_data.cols(); }

	Eigen::Vector2i size() const { return { ref_host_data.rows(), ref_host_data.cols() }; }
	eigen_matrix (const eigen_matrix& other) : eigen_matrix (other.size() ) {}

	int setZero (bool wait = false) { UNUSED(wait); ref_host_data.setZero(); return 1; }
	int setOnes (bool wait = false) { UNUSED(wait); ref_host_data.setOnes(); return 1; }

	~eigen_matrix() { /* nothing */ }
	void alloc_device_mem() { /* nothing */ }
	void free_device_mem() { /* nothing */ }
	void sync_device() { /* nothing */ }
	void sync_host() { /* nothing */ }

};

#endif