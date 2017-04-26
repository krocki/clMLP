/*
* @Author: kmrocki
* @Date:   2016-02-24 10:20:09
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-26 14:16:23
*/

#ifndef __IMPORTER__
#define __IMPORTER__

#include <fstream>

//set Matrix implementation
#include <opencl/cl_matrix.h>

typedef struct {

	cl_matrix<float> x; 	//inputs
	cl_matrix<int> yi; 		//outputs int
	cl_matrix<float> y1; 	//outputs 1-K

} datapoints;

class MNISTImporter {

  public:

	static datapoints importFromFile (cl_ctx& ctx, const char *filename, const char *labels_filename, size_t N ) {

		const size_t offset_bytes = 16;
		const size_t offset_bytes_lab = 8;
		const size_t w = 28;
		const size_t h = 28;
		size_t n_classes = 10;

		datapoints d;

		d.x = cl_matrix<float>(&ctx, {w * h, N});
		d.yi = cl_matrix<int>(&ctx, {1, N});
		d.y1 = cl_matrix<float>(&ctx, {n_classes, N});

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
						d.x.ref_host_data(i, allocs) = ( float ) ( ( uint8_t ) buffer[i] ) / 255.0f;

					d.yi.ref_host_data(0, allocs) = ( unsigned int ) buffer_lab;
					d.y1.ref_host_data.col(allocs) = encoding.col(( unsigned int ) buffer_lab);

					allocs++;

				}

			}

			printf ( "\nSync datapoints to device...\n" );

			d.x.sync_device();
			d.yi.sync_device();
			d.y1.sync_device();

			printf ( "Finished.\n" );

			infile.close();
			labels_file.close();

		} else

			printf ( "Oops! Couldn't find file %s or %s\n", filename, labels_filename );

		return d;

	}

};

#endif

