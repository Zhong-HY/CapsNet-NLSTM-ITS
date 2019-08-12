# CapsNet-NLSTM-ITS
A framework for network-level traffic forecasting is also proposed by sequentially connecting CapsNet and NLSTM.An experiment on a Beijing transportation network with 278 links shows that the proposed framework with the capability of capturing complicated spatiotemporal traffic patterns outperforms multiple state-of-the-art traffic forecasting baseline models.   

The data we use: "Train_data.7z","Test_data.7z","goal_test.7z","goal_train.7z"   
You can get our submitted paper from [Forecasting Transportation Network Speed Using Deep
Capsule Networks with Nested LSTM Models](https://arxiv.org/ftp/arxiv/papers/1811/1811.04745.pdf)
# Model structure
The input has three dimensions, where the first two dimensions represent the resolution of the input image, and the last dimension indicates the amount of channel of the input image  
![Model structure](https://github.com/Zhong-HY/CapsNet-NLSTM-ITS/blob/master/structure.png)
# Transportation network Representation
The traffic state of a roadway link in a road network is defined by the average speed of vehicles that travel on that link.To learn the traffic as an image, the average speed of each link is projected in the road network combined with a GIS map to establish the spatial correspondence between the links and traffic states.  
![Transportation networkRepresentation](https://github.com/Zhong-HY/CapsNet-NLSTM-ITS/blob/master/Transportation%20network%20Representation.png)
# gridding process
First, the road network is segmented by grids with a size of 0.0001° × 0.0001° (latitude and longitude), which guarantees that the two links on a road with opposite directions can be separated into different grids in the studied area. Subsequently, the value of each grid is determined on the basis of the speed of links using the following criteria: if no link passes through the grid area, then the value is zero; if only one link passes through the grid area, then the value is the speed of this link; if multiple links pass through the same grid area, then we assign their average speed to the grid.
![gridding process](https://github.com/Zhong-HY/CapsNet-NLSTM-ITS/blob/master/Schematic%20of%20the%20gridding%20process.png)
# Acknowledgements
 The implementation of CapsNet heavily derived from [Github.com/XifengGuo/CapsNet-Keras](Github.com/XifengGuo/CapsNet-Kera)
