import soundfile as sf
import glob

stems = sorted(glob.glob("/data/slakh2100_flac_redux/train/Track00451/stems/*.flac"))
mix_y = None
for stem in stems:
    y, sr = sf.read(stem)
    if mix_y is None:
        mix_y = y
    else:
        mix_y += y
sf.write("mix_test.flac", mix_y, sr, "PCM_24")

y, sr = sf.read("/data/slakh2100_flac_redux/train/Track00451/mix.flac")
sf.write("mix_gt.flac", y, sr, "PCM_24")
print(y.shape, sr)