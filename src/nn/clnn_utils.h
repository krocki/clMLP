/*
* @Author: kmrocki
* @Date:   2016-02-24 10:47:03
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 21:31:55
*/

#ifndef __CLNN_UTILS_H__
#define __CLNN_UTILS_H__

#include <utils.h>

//set Matrix & Vector implementation
#include <Eigen/Dense>

#include <io/import.h>
#include <iostream>
#include <random>

float cross_entropy ( Eigen::MatrixXf &error, Eigen::MatrixXf &predictions, Eigen::MatrixXf &targets ) {

	float ce = 0.0f;
	error.resize(predictions.rows(), predictions.cols());

	//check what has happened and get information content for that event
	error.array() = -predictions.unaryExpr ( std::ptr_fun ( ::logf ) ).array() * targets.array();
	ce = error.sum();

	return ce;
}

float mse (Eigen::MatrixXf &error, Eigen::MatrixXf &yhat, Eigen::MatrixXf &y ) {

	float mse = 0.0;
	error.resize(y.rows(), y.cols());
	error.array() = y.array() - yhat.array();
	error.array() = 2.0f * error.array() * error.array();
	mse = error.sum();

	return mse;
}

//generate an array of random numbers in range
void matrix_randi ( Eigen::VectorXi &m, int range_min, int range_max ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::uniform_int_distribution<> dis ( range_min, range_max );

	for ( int i = 0; i < m.rows(); i++ )
		m ( i ) = ( float ) dis ( mt );

}

void matrix_randi ( Eigen::MatrixXi &m, int range_min, int range_max ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::uniform_int_distribution<> dis ( range_min, range_max );

	for ( int i = 0; i < m.size(); i++ )
		m ( i ) = ( float ) dis ( mt );

}

//generate an array of random numbers in range
void matrix_rand ( Eigen::MatrixXf &m, float range_min, float range_max ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::uniform_real_distribution<> randf ( range_min, range_max );

	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m ( i, j ) = randf ( mt );

	}

}

void matrix_randn ( Eigen::MatrixXf &m, float mean, float stddev ) {

	std::random_device rd;
	std::mt19937 mt ( rd() );
	std::normal_distribution<> randn ( mean, stddev );

	for ( int i = 0; i < m.rows(); i++ ) {
		for ( int j = 0; j < m.cols(); j++ )
			m ( i, j ) = randn ( mt );
	}

}

void linspace ( Eigen::VectorXi &m, int range_min, int range_max ) {

	// not really linspace - fixed increment = 1, TODO - fix, use range_max
	UNUSED ( range_max );

	for ( int i = 0; i < m.rows(); i++ )
		m ( i ) = ( float ) ( range_min + i );

}

void make_batch ( Eigen::MatrixXf &batch, const Eigen::MatrixXf &data, const Eigen::VectorXi &random_numbers ) {

	// TODO: this crashes when batch_size is > than data[], not checking the bounds, for example , batch_size = 16, data = 5000, 5000 % 16 != 0, will access 5001 and crash
	size_t batch_size = random_numbers.rows();

	for ( size_t i = 0; i < batch_size; i++ )

		batch.col ( i ) = data.col ( random_numbers ( i ) );


}

void mix ( Eigen::MatrixXf &a, const Eigen::MatrixXf &b, const Eigen::MatrixXi &random_numbers ) {

	for ( int i = 0; i < random_numbers.size(); i++ )
		if ( random_numbers ( i ) == 1 ) a.col ( i ) = b.col ( i );

}

void mix ( Eigen::MatrixXf &a, const Eigen::MatrixXf &b, const Eigen::VectorXi &random_numbers ) {

	for ( int i = 0; i < random_numbers.rows(); i++ )
		if ( random_numbers ( i ) == 1 ) a.col ( i ) = b.col ( i );

}

void mix ( Eigen::VectorXf &a, const Eigen::VectorXf &b, const Eigen::VectorXi &random_numbers ) {

	for ( int i = 0; i < random_numbers.rows(); i++ )
		if ( random_numbers ( i ) == 1 ) a.col ( i ) = b.col ( i );

}

void make_batch ( Eigen::MatrixXf &batch, const datapoints &data, const Eigen::VectorXi &random_numbers ) {

	// TODO: this crashes when batch_size is > than data[], not checking the bounds, for example , batch_size = 16, data = 5000, 5000 % 16 != 0, will access 5001 and crash
	size_t batch_size = random_numbers.rows();

	for ( size_t i = 0; i < batch_size; i++ )

		batch.col ( i ) = data.x.col(random_numbers ( i ));


}

void make_batch ( Eigen::MatrixXf &batch, const datapoints &data, const Eigen::MatrixXi &random_numbers ) {

	// TODO: this crashes when batch_size is > than data[], not checking the bounds, for example , batch_size = 16, data = 5000, 5000 % 16 != 0, will access 5001 and crash
	size_t batch_size = random_numbers.cols();

	for ( size_t i = 0; i < batch_size; i++ )

		batch.col ( i ) = data.x.col(random_numbers ( 0, i ));


}

void make_targets ( Eigen::MatrixXf &targets, const datapoints &data, Eigen::VectorXi &random_numbers ) {

	size_t batch_size = random_numbers.rows();

	for ( size_t i = 0; i < ( size_t ) batch_size; i++ )

		targets.col ( i ) = data.y1.col ( random_numbers ( i ) );


}

Eigen::VectorXi colwise_max_index ( Eigen::MatrixXf &m ) {

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

size_t count_zeros ( Eigen::VectorXi &m ) {

	size_t zeros = 0;

	for ( int i = 0; i < m.rows(); i++ ) {

		bool isZero = m ( i ) == 0;

		zeros += isZero;
	}

	return zeros;

}

size_t count_correct_predictions ( Eigen::MatrixXf &p, Eigen::MatrixXf &t ) {

	Eigen::VectorXi predicted_classes = colwise_max_index ( p );
	Eigen::VectorXi target_classes = colwise_max_index ( t );
	Eigen::VectorXi correct = ( target_classes - predicted_classes );

	return count_zeros ( correct );
}

#ifdef USE_BLAS
// c = a * b
void BLAS_mmul ( Eigen::MatrixXf &__restrict c, Eigen::MatrixXf &__restrict a,
                 Eigen::MatrixXf &__restrict b, bool aT, bool bT ) {

	enum CBLAS_TRANSPOSE transA = aT ? CblasTrans : CblasNoTrans;
	enum CBLAS_TRANSPOSE transB = bT ? CblasTrans : CblasNoTrans;

	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = aT ? a.rows() : a.cols();

	float alpha = 1.0f;
	float beta = 1.0f;

	size_t lda = aT ? K : M;
	size_t ldb = bT ? N : K;
	size_t ldc = M;

	cblas_sgemm ( CblasColMajor, transA, transB, M, N, K, alpha,
	              a.data(), lda,
	              b.data(), ldb, beta, c.data(), ldc );

	flops_performed += 2 * M * N * K;
	bytes_read += ( a.size() + b.size() ) * sizeof ( dtype );
}
#endif /* USE_BLAS */

#endif