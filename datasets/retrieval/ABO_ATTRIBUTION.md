# Amazon Berkeley Objects retrieval subset

Source data and images: Amazon.com, Amazon Berkeley Objects (ABO).
Dataset construction: Collins et al., *ABO: Dataset and Benchmarks for Real-World
3D Object Understanding*, CVPR 2022. Source registry:
https://registry.opendata.aws/amazon-berkeley-objects/

The official dataset is licensed CC BY 4.0. The source licence is retained in
ABO_LICENSE.txt. This selection uses the official small (maximum256pixel) images;
no generated or altered product image is presented as real source data.

Sampling frame is only listings_0.json.gz (9,232 listings), not all147,702 ABO
products. Seed8410 selects600 distinct products round-robin across13 exact fashion
product types, with representation constrained by that shard's available classes.
It covers footwear, bags/wallets, hats and jewellery, without apparel coverage.
The metadata/images hashes and actual selected counts are retained in
environment/abo-retrieval-acquisition.json. The initial download encountered a TLS
timeout; already acquired images were reused for the successful retry.

The120 query texts are AI-assisted, authored controlled hypothetical shopping
requests:12 product families crossed with10 predeclared attribute/use-case
descriptions. They are not observed customer requests. No selected model ranking
was consulted in their construction. Human relevance judgements are absent;
qrels stay blank and all corresponding relevance metrics remain missing until
independent assessment. Annotation candidates use a seeded model-independent
random pool of30 products/query; unjudged pool coverage must be reported.

The acquisition script records official object paths and local SHA256s. It never
downloads the entire ABO image archive. To reproduce this exact sampling frame,
use scripts/acquire_abo_retrieval.py and verify the retained metadata hashes;
do not silently accept a changed source object.
