# Execution

## Video Processing
Brackish:
```
py src/track_video.py --video data/brackish/dataset/videos/fish-school/2019-03-19_17-07-53to2019-03-19_17-08-34_1.avi        
```

F4K:
```
py src/track_video.py --video data/f4k/gt_122.mp4
```

## Performance
Brackish:
```
py src/evaluate_tracker.py --gt data/brackish/annotations/annotations_AAU/test.csv --ts phase3_outputs/2019-03-19_17-07-53to2019-03-19_17-08-34_1_results.txt --video-name 2019-03-19_17-07-53to2019-03-19_17-08-34_1
```

F4K:
```
py src/evaluate_f4k.py --gt-xml data/f4k/gt_122.xml --ts-txt phase3_outputs/gt_122_results.txt
```