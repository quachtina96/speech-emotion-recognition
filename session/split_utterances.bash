#!/bin/bash

# Split utterance IDs into smaller batches so that they can be processed in the cluster.

session_number=$1

#mkdir session${session_number}
#split -l 50 session${session_number}_utterance.txt session_${session_number}
mv session_${session_number}a* session${session_number}
mv session_${session_number}b* session${session_number}
