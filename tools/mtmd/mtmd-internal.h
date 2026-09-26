#pragma once

#include "mtmd.h"

#include <string>
#include <vector>

// !!! Internal header, to be used by mtmd and its unit tests only !!!

#define MTMD_INTERNAL_HEADER

// bitmap is null for text parts
struct mtmd_input_part {
    std::string text;
    const mtmd_bitmap * bitmap;
};

// [QWEN_VIDEO] merged parts are erased from `parts`, so one group always maps to one part
std::vector<std::vector<const mtmd_bitmap *>> mtmd_group_mergeable_bitmaps(std::vector<mtmd_input_part> & parts, int n_merge);

// result of comparing the text model identity against the mmproj identity.
// `status` is "ok" when both basenames are present and equal, "mismatch" when both are
// present and differ, and "unknown" when either side is missing its basename.
struct mtmd_mmproj_identity {
    const char * status;
    std::string text_basename;
    std::string mmproj_basename;
};

mtmd_mmproj_identity mtmd_mmproj_identity_result(const std::string & text_basename, const std::string & mmproj_basename);
