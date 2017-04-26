/*
* @Author: kmrocki
* @Date:   2016-02-24 10:47:03
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-26 16:18:31
*/

#ifndef __CLNN_UTILS_H__
#define __CLNN_UTILS_H__

#include <utils.h>
#include <opencl/cl_functions.h>

//set Matrix & Vector implementation
#include <Eigen/Dense>

#include <io/import.h>
#include <iostream>
#include <random>

float cross_entropy_host ( Eigen::MatrixXf &error, Eigen::MatrixXf &predictions, Eigen::MatrixXf &targets ) {

	float ce = 0.0f;

	//check what has happened and get information content for that event
	error.array() = -predictions.unaryExpr ( std::ptr_fun ( ::logf ) ).array() * targets.array();
	ce = error.sum();

	return ce;
}

//generate an array of random numbers in range
void matrix_randi_host ( Eigen::MatrixXi &m, int range_min, int range_max ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::uniform_int_distribution<> dis ( range_min, range_max );

	for ( int i = 0; i < m.size(); i++ )
		m ( i ) = ( float ) dis ( mt );

}

void matrix_randn_host ( Eigen::MatrixXf &m, float mean, float stddev ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::normal_distribution<> randn ( mean, stddev );

	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m ( i, j ) = randn ( mt );
	}

}

void linspace_host ( Eigen::MatrixXi &m, int range_min, int range_max ) {

	// not really linspace - fixed increment = 1, TODO - fix, use range_max
	UNUSED ( range_max );

	for ( int i = 0; i < m.rows(); i++ )
		m ( i, 0 ) = ( float ) ( range_min + i );

}

Eigen::VectorXi colwise_max_index_host ( Eigen::MatrixXf &m ) {

	Eigen::VectorXi indices ( m.cols() );

	for ( size_t i = 0; i < ( size_t ) m.cols(); i++ ) {

		float current_max_val;
		int index;

		for ( size_t j = 0; j < ( size_t ) m.rows(); j++ ) {

			if ( j == 0 || m ( j, i ) > current_max_val ) {

				index = j;
				current_max_val = m ( j, i );
			}

			indices ( i ) = index;

		}
	}

	return indices;
}

size_t count_zeros_host ( Eigen::VectorXi &m ) {

	size_t zeros = 0;

	for ( int i = 0; i < m.rows(); i++ ) {

		bool isZero = m ( i ) == 0;

		zeros += isZero;
	}

	return zeros;

}

size_t count_correct_predictions_host ( Eigen::MatrixXf &p, Eigen::MatrixXf &t ) {

	Eigen::VectorXi predicted_classes = colwise_max_index_host ( p );
	Eigen::VectorXi target_classes = colwise_max_index_host ( t );
	Eigen::VectorXi correct = ( target_classes - predicted_classes );

	return count_zeros_host ( correct );
}

#endif