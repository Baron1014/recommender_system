# recommender_system
RecSys of state-of-the-art 

## performance
<table>
  <tr>
    <td></td>
    <th colspan="3"><CENTER>MovieLens</th>
    <th colspan="3"><CENTER>Yelp</th>
    <th colspan="3"><CENTER>Douban Book</th>
  </tr>
  <tr>
    <th>Model</th>
    <th>RMSE</th>
    <th>Recall@10</th>
    <th>NDCG@10</th>
    <th>RMSE</th>
    <th>Recall@10</th>
    <th>NDCG@10</th>
    <th>RMSE</th>
    <th>Recall@10</th>
    <th>NDCG@10</th>
  </tr>
  <tr>
    <td> UCF-s <td> 1.1859 <td> .5325 <td> .7979 <td> <b>1.2211</b> <td> .0584 <td> .0940 <td> 0.9581 <td> .1747 <td> .3144
  </tr>
  <tr>
  <td>UCF-p <td> 1.1891 <td> .5237 <td>.7902 <td> 1.2863 <td> .0585 <td> .0933 <td> 0.9644 <td> .1729 <td> .3100
  </tr>
  <tr>
    <td>ICF-s <td> 1.1316 <td> .0037 <td> .5413 <td> 1.2735 <td> .0003 <td> .0408 <td> 0.8248 <td> .0001 <td> .3104 
  </tr>
  <tr>
    <td>ICF-p <td> 1.1116 <td> .0037 <td> .5185 <td> 1.2474 <td>.0003 <td> .0395 <td> 0.8136 <td> .0001 <td> .3092 
  </tr>
  <tr>
    <td>MF <td> 0.9417 <td> .0000 <td> .2410 <td> 1.5662 <td> .0001 <td> .1002 <td> <b>0.7659</b> <td> .0002 <td> .1431
  </tr>
  <tr>
    <td>FM <td> 0.7834 <td> .0043 <td> .2640 <td> 1.2995 <td> .0006 <td> .1090 <td> 0.8581 <td> .0048 <td> .1534
  </tr>
  <tr>
    <td> BPR-MF <td> <b>0.5530</b> <td> .0004 <td> .2492 <td colspan="6" rowspan="4" align="center"> out of memory
  </tr>
  <tr>
    <td>BPR-FM <td> 0.7124 <td> .0421 <td>.2991 
  </tr>
  <tr>
    <td>GBDT+LR <td> 1.3435 <td> .0172 <td> .2679 
  </tr>
  <tr>
    <td>XGB+LR <td> 1.3514 <td> .0172 <td> .2679 
  </tr>
  <tr>
   <td> FNN <td> 1.5966 <td> .4782 <td> .8633 <td> 1.5123 <td> .7921 <td> .7213 <td> 1.6594 <td> .6500 <td> .8765
  </tr>
   <td> IPNN <td> 1.6167 <td> .4767 <td> .8618 <td> 1.5209 <td> .7930 <td> .7335 <td> 1.6772 <td> .6481 <td> .8773
  <tr>
   <td> OPNN <td> 1.6056 <td> .4781 <td> .8623 <td> 1.5304 <td> .7851 <td> .7273 <td> 1.6768 <td> .6475 <td> .8770
  </tr>
  <tr>
    <td>PIN <td> 1.6145 <td> .4771 <td> .8601 <td> 1.5231 <td> .7884 <td> .7300 <td> 1.6825 <td> .6477 <td> .8780
  </tr>
  <tr>
   <td> CCPM <td> 1.5746 <td> .4778 <td> .8689 <td> 1.5114 <td> .7969 <td> .7223 <td> 1.6600 <td> .6461 <td> .8806
  </tr>
  <tr>
    <td>WD <td> 1.5900 <td> .4791 <td> .8633 <td> 1.5132 <td> .7914 <td> .7217 <td> 1.6616 <td> .6493 <td> .8748
  </tr>
  <tr>
    <td>DCN <td> 1.5916 <td> .4793 <td> .8660 <td> 1.5137 <td> .7893 <td> .7217 <td> 1.6602 <td> .6495 <td> .8739 
  </tr>
  <tr>
   <td> NFM <td> 1.5480 <td> .4767 <td> .8729 <td> 1.5193 <td> .8009 <td> .7272 <td> 1.6410 <td> .6501 <td> .8877 
  </tr>
  <tr>
    <td>DeepFM <td> 1.5940 <td> .4794 <td> .8652 <td> 1.5130 <td> .7903 <td> .7218 <td> 1.6556 <td> .6499 <td> .8756 
  </tr>
  <tr>
    <td>AFM <td> 1.4988 <td> .4884 <td> .8865 <td> 1.3584 <td> .8139 <td> .7742 <td> 1.6169 <td> .6523 <td> .8910
  </tr>
  <tr>
   <td> xDeepFM <td> 1.6086 <td> .4794 <td> .8646 <td> 1.5126 <td> .7940 <td> .7245 <td> 1.6534 <td> .6501 <td> .8750
  </tr>
  <tr>
   <td> DIN <td> 1.9667 <td> .4884 <td> .8832 <td colspan="6"  align="center">out of memory
  </tr>
  <tr>
    <td>Proposed <td> 1.4468 <td> <b>.4945</b> <td> <b>.8992</b> <td> 1.3513 <td> <b>.8327</b> <td> <b>.7843</b> <td> 1.6042 <td> <b>.6555</b> <td> <b>.8938</b> 

  </tr>
 </table>
  
