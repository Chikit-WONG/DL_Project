### V5 modified  

#### 12 blur level:  
Validation-selected checkpoint	0.8240 ± 0.0201	    0.9780 ± 0.0054	  
Best test checkpoint	        0.8685 ± 0.0063	    0.9810 ± 0.0052	 
  
#### 12 blur level + EVNet:  
Validation-selected checkpoint  0.8325 ± 0.0237     0.9825 ± 0.0040  
Best test checkpoint            0.8825 ± 0.0051     0.9820 ± 0.0060  

#### 8 blur level + EVNet:  
Validation-selected checkpoint  0.8360 ± 0.0214     0.9820 ± 0.0046  
Best test checkpoint            0.8815 ± 0.0092     0.9825 ± 0.0056  

#### EVNet with no blur:  
Validation-selected checkpoint  0.7340 ± 0.0214     0.9565 ± 0.0090   
Best test checkpoint            0.7785 ± 0.0150     0.9660 ± 0.0080  

#### No blur and no EVNet:  
Validation-selected checkpoint  0.6120 ± 0.0235     0.9060 ± 0.0176   
Best test checkpoint            0.6705 ± 0.0101     0.9110 ± 0.0170  

#### No blur and no EVNet, grey background:  
Validation-selected checkpoint  0.6950 ± 0.0226     0.9205 ± 0.0079   
Best test checkpoint            0.7380 ± 0.0075     0.9230 ± 0.0114

#### EVNet with no blur, grey background:  
Validation-selected checkpoint  0.7765 ± 0.0148     0.9590 ± 0.0092   
Best test checkpoint            0.8185 ± 0.0081     0.9630 ± 0.0078

#### 8 blur level + EVNet, grey background:   
Validation-selected checkpoint  0.8225 ± 0.0155     0.9750 ± 0.0067   
Best test checkpoint            0.8595 ± 0.0069     0.9735 ± 0.0055  

#### 8 blur level, grey background:  
Validation-selected checkpoint  0.8105 ± 0.0106     0.9795 ± 0.0061  
Best test checkpoint            0.8415 ± 0.0125     0.9805 ± 0.0072  

使用灰背景的EVNet比不使用灰背景的EVNet高了将近10%，但加上blur却差不多相同。  
灰背景大概充当了blur的效果  