### V5 modified  

#### 12 blur level:  
Validation-selected checkpoint	0.8240 ± 0.0201	    0.9780 ± 0.0054	  
Best test checkpoint	        0.8685 ± 0.0063	    0.9810 ± 0.0052	 

#### 8 blur level:  
Validation-selected checkpoint	0.8280 ± 0.0234	    0.9780 ± 0.0051	 
Best test checkpoint            0.8705 ± 0.0085     0.9790 ± 0.0049  

#### 8 blur level + EVNet:  
Validation-selected checkpoint  0.8530 ± 0.0081     0.9845 ± 0.0035  
Best test checkpoint            0.8890 ± 0.0107     0.9855 ± 0.0035  

#### 8 blur level + EVNet, Hungarian:  
Validation-selected checkpoint  0.9675 ± 0.0078     0.9900 ± 0.0045  
Best test checkpoint            0.9910 ± 0.0062     0.9965 ± 0.0032  
Final epoch                     0.9650 ± 0.0132     0.9905 ± 0.0065  

#### EVNet with no blur:  
Validation-selected checkpoint  0.7075 ± 0.0211     0.9350 ± 0.0095  
Best test checkpoint            0.7565 ± 0.0084     0.9415 ± 0.0147

#### No blur and no EVNet:  
Validation-selected checkpoint  0.6120 ± 0.0235     0.9060 ± 0.0176   
Best test checkpoint            0.6705 ± 0.0101     0.9110 ± 0.0170  

#### No blur and no EVNet, grey background:  
Validation-selected checkpoint  0.6950 ± 0.0226     0.9205 ± 0.0079   
Best test checkpoint            0.7380 ± 0.0075     0.9230 ± 0.0114

#### 8 blur level, grey background:  
Validation-selected checkpoint  0.8105 ± 0.0106     0.9795 ± 0.0061  
Best test checkpoint            0.8415 ± 0.0125     0.9805 ± 0.0072  

使用灰背景的EVNet比不使用灰背景的EVNet高了将近10%，但加上blur却差不多相同。  
灰背景大概充当了blur的效果  
现在所有EVNet组默认是修复后的，没有修复的EVNet的数据在v5_log  