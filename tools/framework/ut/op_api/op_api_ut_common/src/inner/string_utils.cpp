#include <algorithm>
#include "op_api_ut_common/inner/string_utils.h"

using namespace std;

vector<string> Split(const string &s, const string &spliter) {
  vector<string> strings;
  size_t i = 0;
  while (i < s.size()) {
    auto pos = s.find_first_of(spliter, i);
    if (pos == string::npos) {
      strings.emplace_back(s, i);
      break;
    } else {
      strings.emplace_back(s, i, pos - i);
      i = pos + spliter.size();
    }
  }

  return strings;
}

string &Ltrim(string &s) {
  (void) s.erase(s.begin(),
                 find_if(s.begin(), s.end(),
                         [](const char c) {
                           return static_cast<bool>(isspace(static_cast<unsigned char>(c)) == 0);
                         }));
  return s;
}

string &Rtrim(string &s) {
  (void) s.erase(find_if(s.rbegin(), s.rend(),
                         [](const char c) {
                           return static_cast<bool>(isspace(static_cast<unsigned char>(c)) == 0);
                         }).base(),
                 s.end());
  return s;
}

string Trim(string &s) {
  return Ltrim(Rtrim(s));
}