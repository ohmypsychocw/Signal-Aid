#!/usr/bin/env bash
# Publish the replay demo (one static file) to Amazon S3 behind Amazon CloudFront.
#   bash scripts/deploy_web.sh <bucket-name>
# First time: create the bucket and a CloudFront distribution with Origin Access
# Control pointing to it (console: CloudFront > Create distribution > S3 origin).
set -euo pipefail
BUCKET="${1:?usage: deploy_web.sh <bucket-name>}"
aws s3 cp web/dist/index.html "s3://${BUCKET}/index.html" \
  --content-type "text/html; charset=utf-8" --cache-control "max-age=60"
echo "Uploaded. Open your CloudFront URL, for example https://dxxxxxxxx.cloudfront.net/"
