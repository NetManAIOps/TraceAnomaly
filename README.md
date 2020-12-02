# TraceAnomaly
Detecting anomalous traces of microservice system.

# Paper
Ping Liu, Haowen Xu, Qianyu Ouyang, Rui Jiao, Zhekang Chen, Shenglin Zhang, Jiahai Yang, Linlin Mo, Jice Zeng, Wenman Xue, Dan Pei. Unsupervised Detection of Microservice Trace Anomalies through Service-Level Deep Bayesian Networks". 31th International Symposium on Software Reliability Engineering (ISSRE). IEEE, 2020

paper download（论文下载）：https://netman.aiops.org/wp-content/uploads/2020/09/%E5%88%98%E5%B9%B3issre.pdf
## Dependencies

Python == 3.6

```shell
pip install -r requirements.txt
```

### Docker Image
TraceAnomaly can be run directly in the Docker image: **silence1990/docker_for_traceanomaly:latest**

```bash
docker pull silence1990/docker_for_traceanomaly:latest
```
## Dataset
Training set: train_ticket/train.zip

Test normal traces: train_ticket/test_normal.zip

Test anomalous traces: train_ticket/test_abnormal.zip
## Usage

```shell
./run.sh
```

## Comparison of Learning Distribution

![image](https://github.com/NetManAIOps/TraceAnomaly/blob/master/traceanomaly/performance.png)

