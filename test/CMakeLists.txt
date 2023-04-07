# Copyright 2020 Mikhail Katliar. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

add_subdirectory(blast)

if (BLAST_WITH_BLASFEO)
    add_subdirectory("blasfeo")
endif ()