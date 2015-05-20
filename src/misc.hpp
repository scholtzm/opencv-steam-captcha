#ifndef _MISC_H_
#define _MISC_H_

using namespace std;

/**
 * Functions to translate special characters
 * @, % and & to "at", "pct" and "and"
 * and vice versa.
 */
string specialCharToAlias(string s);
string aliasToSpecialChar(string s);

/**
 * Creates folder structure in the output folder.
 */
bool createFolderStructure(string output_folder, string folders);

#endif