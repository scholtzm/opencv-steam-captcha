#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
namespace fs = boost::filesystem;

string specialCharToAlias(string s) {
    boost::replace_all(s, "@", "at");
    boost::replace_all(s, "%", "pct");
    boost::replace_all(s, "&", "and");
    
    return s;
}

string aliasToSpecialChar(string s) {
    boost::replace_all(s, "at", "@");
    boost::replace_all(s, "pct", "%");
    boost::replace_all(s, "and", "&");
    
    return s;
}

bool createFolderStructure(string output_folder, string folders) {
    fs::path folder(output_folder);
    if(!exists(folder) && !fs::create_directory(folder))
        return false;
    
    for(int i = 0; i < folders.length(); i++) {
        string letter(1, folders[i]);
        
        string sub = specialCharToAlias(letter);
        
        fs::path dir(output_folder + sub);
        
        if(exists(dir))
            continue;
        
        if(!fs::create_directory(dir))
            return false;
    }
    
    return true;
}