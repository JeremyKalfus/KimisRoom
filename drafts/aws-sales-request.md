Paste into https://aws.amazon.com/contact-us/sales-support/ (or live chat: https://pages.awscloud.com/live-chat-contact-us.html). Support case 179039677400616 said further GPU quota goes through sales.

---

Subject: GPU quota for a research project on AWS Activate credits (account 522540831020)

Hello,

I'm requesting 192 vCPUs of Running On-Demand P instances in us-east-2 (quota L-417A185B) so I can run one p5en.48xlarge. Support case 179039677400616 approved 64 vCPUs and said any further increase has to go through the account or sales team.

Workload: an interpretability study testing whether an open-weight model's self-reports about emotion track its internal state. The model is Kimi K2.6 (1T parameters, ~600 GB of native INT4 weights). Its validated deployment needs 8x H200 on one node, so 64 vCPUs fits no instance that can hold it. Each run is a batch job (load weights, score the prompts, write results to S3, terminate), not a persistent service. Total use is about 30-50 instance-hours over the next two months.

Billing: covered by the account's AWS Activate credit (YC University Program, $10,000, valid to 2028-09-30). On-demand only; no Reserved Instances, Savings Plans or Marketplace.

Meanwhile I'm using the 64 vCPUs of G quota already granted (g6e.12xlarge). I also have a pending SageMaker request for 1 x ml.p5en.48xlarge for training job usage in us-east-2 (quota L-1E48384D). Either route works, and I only need one of them.

Results will go to David Chalmers (NYU), who encouraged the project, and will be published openly.

Thank you,
Jeremy Kalfus
