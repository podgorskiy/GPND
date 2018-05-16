import train_AAE
import novelty_detector

fold = 0

for i in range(10):
    train_AAE.main(fold, [i], 9)
    novelty_detector.main(fold, [i], 9)
