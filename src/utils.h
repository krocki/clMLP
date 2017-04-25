/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-03-03 15:22:47
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 16:29:14
*/

#ifndef __UTIL_MAIN_H__
#define __UTIL_MAIN_H__

// to surpress warnings
#define UNUSED(...) [__VA_ARGS__](){};

#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdarg.h>  // For va_start, etc.
#include <memory>    // For std::unique_ptr
#include <iostream>
#include <cmath>
#include <cfloat>

#include <Eigen/Dense>

double get_time_diff (const struct timeval* s, const struct timeval* e) {

	// computes time difference between s and e
	struct timeval  diff_tv;

	diff_tv.tv_usec = e->tv_usec - s->tv_usec;
	diff_tv.tv_sec = e->tv_sec - s->tv_sec;

	if (s->tv_usec > e->tv_usec) {
		diff_tv.tv_usec += 1000000;
		diff_tv.tv_sec--;
	}

	return (double) diff_tv.tv_sec + ( (double) diff_tv.tv_usec / 1000000.0);
}

std::string return_current_time_and_date ( const char *format = "%x %X" ) {

	auto now = std::chrono::system_clock::now();
	auto in_time_t = std::chrono::system_clock::to_time_t ( now );

	std::stringstream ss;
	ss << std::put_time ( std::localtime ( &in_time_t ), format );
	return ss.str();
}

template <typename T>
std::string to_string_with_precision ( const T a_value, const int m = 12, const int n = 5 ) {
	std::ostringstream out;
	out << std::fixed << std::setw ( m ) << std::setprecision ( n ) << std::setfill ( '_' ) << a_value;
	return out.str();
}

std::string string_format ( const std::string fmt_str, ... ) {
	int final_n, n = ( ( int ) fmt_str.size() ) * 2; /* Reserve two times as much as the length of the fmt_str */
	std::string str;
	std::unique_ptr<char[]> formatted;
	va_list ap;
	while ( 1 ) {
		formatted.reset ( new char[n] ); /* Wrap the plain char array into the unique_ptr */
		strcpy ( &formatted[0], fmt_str.c_str() );
		va_start ( ap, fmt_str );
		final_n = vsnprintf ( &formatted[0], n, fmt_str.c_str(), ap );
		va_end ( ap );
		if ( final_n < 0 || final_n >= n )
			n += abs ( final_n - n + 1 );
		else
			break;
	}
	return std::string ( formatted.get() );
}

// taken from
// http://stackoverflow.com/questions/83439/remove-spaces-from-stdstring-in-c
std::string delUnnecessary(std::string & str) {
	int size = str.length();

	for (int j = 0; j <= size; j++) {
		for (int i = 0; i <= j; i++) {
			if (str[i] == ' ' && str[i + 1] == ' ') {
				str.erase(str.begin() + i);

			} else if (str[0] == ' ') {
				str.erase(str.begin());

			} else if (str[i] == '\0' && str[i - 1] == ' ') {
				str.erase(str.end() - 1);
			}
		}
	}

	return str;
}

std::string getFirstWord(std::string & str) {
	return str.substr(0, str.find(' '));
}

std::vector<std::string> split ( std::string in ) {
	std::vector<std::string> strings;
	std::istringstream f ( in );
	std::string s;
	while ( getline ( f, s, ' ' ) ) {
		std::cout << s << std::endl;
		strings.push_back ( s );
	}

	return strings;
}

char* readFile(const char* filename, size_t* length) {

	// locals
	FILE*           file = NULL;
	size_t          len;

	// open the  file
	file = fopen (filename, "rb");

	if (file == 0) {
		printf ("Can't open %s\n", filename);
		return NULL;
	}

	// get the length
	fseek (file, 0, SEEK_END);
	len = (size_t) ftell (file);
	fseek (file, 0, SEEK_SET);

	// allocate a buffer for the string and read it in
	char* buffer = (char*) malloc (len + 1);
	int ret = (int) fread ((buffer), len, 1, file);

	if (ret == 0) {
		printf ("Can't read source %s\n", filename);
		return NULL;
	}

	// close the file and return the total length of the combined (preamble +
	// source) string
	fclose (file);

	if (length != 0) {
		*length = len;
	}

	buffer[len] = '\0';
	return buffer;
}


// rands
float rand_float ( float mn, float mx ) {

	float r = random() / ( float ) RAND_MAX;
	return mn + ( mx - mn ) * r;
}

Eigen::Matrix4f quat_to_mat ( const Eigen::Quaternionf &q ) {

	Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
	mat.block ( 0, 0, 3, 3 ) = q.matrix();
	return mat;

}

Eigen::Matrix4f translate ( const Eigen::Vector3f &v ) {
	return Eigen::Affine3f ( Eigen::Translation<float, 3> ( v ) ).matrix();
}

Eigen::Quaternionf rotate ( const Eigen::Vector3f &angle, const Eigen::Vector3f &forward, const Eigen::Vector3f &up,
                            const Eigen::Vector3f &right ) {

	Eigen::AngleAxisf roll ( angle[0], forward );
	Eigen::AngleAxisf yaw ( angle[1], up );
	Eigen::AngleAxisf pitch ( angle[2], right );

	return yaw * pitch * roll;
}

void histogram ( Eigen::MatrixXf &m, Eigen::VectorXf *h ) {

	h->setZero();

	// float _min_ = m.minCoeff();
	// float _max_ = m.maxCoeff();

	// float bin_size = (_max_ - _min_ / (float)h->size());

	// std::cout << bin_size;

	float bin_size = 0.2f;
	float min_val = -1.0f;

	// x - 0.8
	// - 0.8 - 0.6
	// - 0.6 - 0.4
	// - 0.4 - 0.2
	// - 0.2 - 0.0
	// 0.0 0.2
	// 0.2 0.4
	// 0.4 0.6
	// 0.6 0.8
	// 0.8 -

	// std::cout << min_val << std::endl;

	for ( size_t ii = 0; ii < m.size(); ii++ ) {

		float val = m ( ii );
		int bin_id = ( val - min_val ) / bin_size;
		if ( bin_id > ( h->size() - 1 ) ) bin_id = h->size() - 1;
		if ( bin_id < 0 ) bin_id = 0;
		h->operator() ( bin_id ) += 1;

	}

	float _max_hist_ = h->maxCoeff();

	for ( size_t ii = 0; ii < h->size(); ii++ )

		h->operator() ( ii ) /= _max_hist_;


}

bool isNaNInf ( float f ) {

	return ( std::isnan ( f ) || std::isinf ( f ) );

}

void checkNaNInf ( Eigen::MatrixXf &m ) {

	m = m.unaryExpr ( [] ( float elem ) { // changed type of parameter
		return ( std::isnan ( elem ) || std::isinf ( elem ) ) ? 0.0f : elem; // return instead of assignment
	} );

}

template<typename T>
void removeSubstrs(std::basic_string<T>& s, const std::basic_string<T>& p) {
	typename std::basic_string<T>::size_type n = p.length();

	for (typename std::basic_string<T>::size_type i = s.find(p);
	        i != std::basic_string<T>::npos;
	        i = s.find(p))
		s.erase(i, n);
}

#endif
