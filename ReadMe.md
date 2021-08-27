Deep Learning Analysis to identify the gender based on the eyes.

---

&nbsp;

### **CLI Arguments**

<pre>
1. --full       : Flag that controls whether to train on reduced or full data (Default: Reduced)

2. --bs         : Batch Size for dataloader creation (Default: 64)

3. --lr         : Learning Rate (Default: 1e-3)

4. --wd         : Weight Decay (Default: 0)

5. --scheduler  : If specified, it requires 2 arguments; patience and eps (Default: Not Specified)

6. --epochs     : Number of training epochs (Default: 10)

7. --early      : Early Stopping Patience (Default: 5)

8. --train-full : Flag that controls whether the entire model must be trained or only the final layer (Default: None)

9. --augment    : Flag that controls whether training data must be augmented (Default: None)

10. --test       : Flag that controls entry into test mode (Default: None)

11. --name      : Name of the image file to be tested
</pre>

&nbsp;

---

**Notes**

1. Research eye detectors
