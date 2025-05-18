#!/bin/bash

# SPDX-FileCopyrightText: 2023 - 2025 Yadong Zeng<zdsjtu@gmail.com> & ZhuXu Li<1246206018@qq.com>
#
# SPDX-License-Identifier: BSD-3-Clause

rm -rf output/*

python3 plt_to_numpyArray.py -v experiment000019 y_velocity 3 yes
