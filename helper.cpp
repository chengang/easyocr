#include <iostream>
#include <sstream>
#include <string>
#include <locale>
#include <boost/filesystem.hpp>

#include "helper.h"

std::string Helper::GetHashPath(std::string basePath, std::string filename) {
    std::locale loc;
    const std::collate<char>& coll = std::use_facet<std::collate<char> >(loc);

    long hash = coll.hash(filename.data(), filename.data()+filename.length());
    int one = std::abs(hash % 977);
    int two = std::abs(hash % 997);
    
    std::stringstream hashPath;
    hashPath << "/" << one << "/" << two;

    boost::filesystem::path p (basePath + hashPath.str());
    boost::filesystem::create_directories(p);

    return hashPath.str();
}
