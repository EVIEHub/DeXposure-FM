"""INTERNAL NOTES: compare cell colours of the original and recomputed Figure 4."""
from pathlib import Path
import subprocess,json
import numpy as np
from PIL import Image
P=Path(__file__).resolve().parent;repo=P.parents[3];dest=P/'figure4_images';dest.mkdir(exist_ok=True)
for name,path in [('paper',repo/'DeXposure_FM_V2/figures/fig_spillover_matrix_example.pdf'),('new',P/'figures/fig4_recomputed.pdf')]:
 subprocess.run(['pdfimages','-png',str(path),str(dest/name)],check=True)
m=json.loads((P/'figure4_matrix.json').read_text());n=len(m['july_input']['sectors']);assert n==15
images=[np.array(Image.open(dest/f'{s}-000.png').convert('RGB')) for s in ['paper','new']]
def cells(z):return np.array([[z[int((i+.5)*z.shape[0]/n),int((j+.5)*z.shape[1]/n)] for j in range(n)] for i in range(n)])
a,b=images;c,d=map(cells,images)
r={'paper_image_shape':a.shape,'new_image_shape':b.shape,'pixel_identical':a.shape==b.shape and np.array_equal(a,b),'sampled_cells':n*n,'cell_rgb_equal':int(np.sum(np.all(c==d,axis=2))),'max_channel_difference':int(np.abs(c.astype(int)-d.astype(int)).max())}
(P/'figure4_raster_comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
