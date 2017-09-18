"""
Illustate the levels in a scale space pyramid.
"""

import visvis as vv

final_scale = 1.0
scale_sampling = 7
scale_levels = 3

iter_factor = 0.5**(1.0/scale_sampling)

finetune_factor = 1.0/(scale_sampling+1)

def calc_scales(smooth_scale):
    
    iters = []
    scales = []
        
    count = 0
    for level in reversed(range(scale_levels)):
        
        # Set (initial) scale for this level
        scale = final_scale * 2**level
        if smooth_scale:
            scale *= 2 * iter_factor
        
        for iter in range(1, scale_sampling+1):
            count += 1
            
            if smooth_scale and level >= scale_levels-1:
                continue
            
            # Store scale
            iters.append(count)
            scales.append(scale)
            
            # Next iteration
            if smooth_scale:
                scale = max(final_scale, scale*iter_factor)
#                 if level==0 and iter > 0:#> 0.5*scale_sampling:
#                     break
    return iters, scales

vv.figure(10); vv.clf()
a = vv.gca()
iters, scales = calc_scales(False); vv.plot(iters, scales, lc='b', mc='b', ms='.')
iters, scales = calc_scales(True);  vv.plot(iters, scales, lc='g', mc='g', ms='.')
a.axis.showGrid = True

vv.use().Run()
