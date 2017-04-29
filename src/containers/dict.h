/*
    @Author: kmrocki@us.ibm.com
    @Date:   2017-04-01 19:33:21
    @Last Modified by:   Kamil Rocki
    @Last Modified time: 2017-04-28 15:17:42

    m = Dict(

        {   "something" , {

            std::make_tuple("W", M, N),
            std::make_tuple("U", X, Y),

        }
    )

    will make a 2 sub-entries "W" and "U" of dimensions M x N and X x Y
    respectively;

    later, m['W'] will return the first matrix and m['U'] the second one
    other things are just implementations of operators and IO
*/

#ifndef __DICT_H__
#define __DICT_H__

#include <map>
#include <vector>
#include <string>

template <typename T>
class Dict {

  public:

	std::vector<T> entries;
	std::string name;
	std::map<std::string, size_t> namemap;
	std::map<size_t, std::string> reverse_namemap;

	Dict<T>() = default;

	Dict<T> (const Dict<T>& other) {
		namemap = other.namemap;
		reverse_namemap = other.reverse_namemap;
		name = other.name;
		entries = other.entries;
	}

	Dict<T>& operator= (const Dict<T>& other) {
		namemap = other.namemap;
		reverse_namemap = other.reverse_namemap;
		name = other.name;
		entries = other.entries;
		return *this;
	}

	template <typename otherType>
	Dict<T>& operator= (const Dict<otherType>& other) {
		namemap = other.namemap;
		reverse_namemap = other.reverse_namemap;
		name = other.name;

		for (size_t i = 0; i < entries.size(); i++)
			entries[i] = other.entries[i];

		return *this;
	}

	T& operator[] (char key) {
		return (*this) [std::string (1, key)];
	}

	T& operator[] (std::string key) {
		if (namemap.find (key) == namemap.end() ) {
			namemap[key] = entries.size();
			reverse_namemap[entries.size()] = key;
			entries.resize (entries.size() + 1);
		}

		return entries[namemap[key]];
	}

	T* ptr (std::string key) {
		if (namemap.find (key) == namemap.end() ) {
			namemap[key] = entries.size();
			reverse_namemap[entries.size()] = key;
			entries.resize (entries.size() + 1);
		}

		return & (entries[namemap[key]]);
	}

	template <class F>
	std::vector<size_t> sorted_idxs (const F& f) {
		return sort_idxs (entries, f);
	}

	template<class Archive>
	void serialize (Archive& archive) {
		archive (name);
		archive (namemap);
		archive (entries);
	}

};

#endif /*__Dict_H__*/
