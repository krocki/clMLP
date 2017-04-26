/*
* @Author: kmrocki
* @Date:   2016-02-24 10:20:09
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 20:43:46
*/

#ifndef __IMPORTER__
#define __IMPORTER__

#include <deque>
#include <fstream>

//set Matrix implementation
#include <Eigen/Dense>
typedef Eigen::VectorXf Vector;

typedef struct {

	Eigen::MatrixXf x; 	//inputs
	Eigen::MatrixXi yi; //outputs int
	Eigen::MatrixXf y1; //outputs 1-K

} datapoints;

class MNISTImporter {

  public:

	static datapoints importFromFile ( const char *filename, const char *labels_filename, size_t N ) {

		const size_t offset_bytes = 16;
		const size_t offset_bytes_lab = 8;
		const size_t w = 28;
		const size_t h = 28;

		datapoints d;

		d.x.resize(w * h, N);
		d.yi.resize(1, N);

		size_t n_classes = 10;
		d.y1.resize(n_classes, N);
		Eigen::MatrixXf encoding = Eigen::MatrixXf::Identity ( n_classes, n_classes );

		char buffer[w * h];
		char buffer_lab;

		size_t allocs = 0;

		std::ifstream infile ( filename, std::ios::in | std::ios::binary );
		std::ifstream labels_file ( labels_filename, std::ios::in | std::ios::binary );

		if ( infile.is_open() && labels_file.is_open() ) {

			printf ( "Loading data from %s", filename );
			fflush ( stdout );

			infile.seekg ( offset_bytes, std::ios::beg );
			labels_file.seekg ( offset_bytes_lab, std::ios::beg );

			while ( !infile.eof() && !labels_file.eof() ) {

				infile.read ( buffer, w * h );
				labels_file.read ( &buffer_lab, 1 );

				if ( !infile.eof() && !labels_file.eof() ) {

					if ( allocs % 1000 == 0 ) {
						putchar ( '.' );
						fflush ( stdout );
					}

					for ( unsigned i = 0; i < w * h; i++ )
						d.x(i, allocs) = ( float ) ( ( uint8_t ) buffer[i] ) / 255.0f;

					d.yi(0, allocs) = ( unsigned int ) buffer_lab;
					d.y1.col(allocs) = encoding.col(( unsigned int ) buffer_lab);

					allocs++;

				}

			}

			printf ( "Finished.\n" );
			infile.close();
			labels_file.close();

		} else

			printf ( "Oops! Couldn't find file %s or %s\n", filename, labels_filename );

		return d;

	}

};

#endif

