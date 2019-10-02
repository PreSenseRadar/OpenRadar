# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from mmwave.radars import ti
from mmwave.tracking import ekf
from mmwave.tracking import GTRACK_visualize
import time
import numpy as np

if __name__ == '__main__':

    tracker = ekf.EKF()
    radar = ti.TI(cli_loc='COM3', data_loc='COM5', mode=1)
    radar._initialize()

    GTRACK_visualize.create()
    while True:
        time.sleep(.1)
        try:
            data = radar.sample()
            
            pc = data['pointCloud2D']
            ranges = pc['range']
            azimuths = pc['azimuth']
            dopplers = pc['doppler']
            dopplers = np.ones_like(dopplers) / 10
            snrs = pc['snr']
            print(len(ranges))
        except:
            continue
        
        if data is not None:
            frame = GTRACK_visualize.get_empty_frame()
            
            tracker.update_point_cloud(ranges, azimuths, dopplers, snrs)
            
            targetDescr, tNum = tracker.step()

            try:
                frame = GTRACK_visualize.update_frame(targetDescr, int(tNum[0]), frame)
            except:
                pass
            try:
                frame = GTRACK_visualize.draw_points(tracker.point_cloud, len(ranges), frame)
            except:
                pass
            if not GTRACK_visualize.show(frame, wait=10):
                break
        else:
            frame = GTRACK_visualize.get_empty_frame()
#            frame = GTRACK_visualize.update_frame(target_desc, 0, frame)
            if not GTRACK_visualize.show(frame, wait=10):
                break

    GTRACK_visualize.destroy()
    radar.close()