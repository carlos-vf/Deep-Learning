# Execution

## Video Processing
Brackish:
```
py src/tracker/track_video.py --video data/brackish/dataset/videos/fish-school/2019-03-19_17-07-53to2019-03-19_17-08-34_1.avi        
```

F4K:
```
py src/tracker/track_video.py --video data/f4k/gt_122.mp4
```

FishTrack2023:
```
py src/tracker/track_video.py --video data\fishtrack23\us_pi_2017_pifsc_mouss_sampler\gindai3\output.mp4
```

## Performance
Brackish:
```
py src/tracker/evaluate_tracker.py --gt data/brackish/annotations/annotations_AAU/test.csv --ts phase3_outputs/2019-03-19_17-07-53to2019-03-19_17-08-34_1_results.txt --video-name 2019-03-19_17-07-53to2019-03-19_17-08-34_1
```

F4K:
```
py src/tracker/evaluate_f4k.py --gt-xml data/f4k/gt_122.xml --ts-txt phase3_outputs/gt_122_results.txt
```

FishTrack2023:
```
py .\src\tracker\evaluate_fishtrack23.py --gt-json "data\fishtrack23\us_pi_2017_pifsc_mouss_sampler\gindai3\annotations.dive.json" --ts-txt "phase3_outputs\output_results.txt"
```