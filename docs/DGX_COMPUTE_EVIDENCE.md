# DGX compute and storage findings

Account: `<user>`
Namespace: `<namespace>`
Tested: 2026-08-21

We wanted to know exactly what this account can request on the cluster before
planning any long training run, so we checked it directly instead of assuming.

## How we did it

Access and pod creation follow the university's `DGX CONNECT.pptx` deck. For the
verification commands and for what the resource names actually mean, we used the
upstream documentation:

- [Kubernetes: Resource Quotas](https://kubernetes.io/docs/concepts/policy/resource-quotas/),
  for namespace `ResourceQuota` objects and how they are enforced at admission.
- [Kubernetes: Scheduling GPUs](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/),
  for extended resources and why GPU counts are fixed when the pod is created.
- [NVIDIA: MIG Support in Kubernetes](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-operator-mig.html),
  for the `nvidia.com/mig-<N>g.<M>gb` naming.
- [NVIDIA: MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/),
  for MIG geometry and the fact that slices stay isolated rather than pooling.

## Nodes

```console
$ kubectl get nodes
NAME                 STATUS   ROLES                  AGE    VERSION
<cluster-node>    Ready    control-plane,master   280d   v1.32.10
<cluster-node>   Ready    worker                 280d   v1.32.10
<cluster-node>   Ready    worker                 280d   v1.32.10
```

## GPU capacity on the workers

```console
$ kubectl describe node <cluster-node> | grep -A20 "Allocatable:"
Allocatable:
  cpu:                        224
  ephemeral-storage:          1723416526453
  memory:                     2113351296Ki
  nvidia.com/gpu:             0
  nvidia.com/mig-1g.18gb:     56
  nvidia.com/mig-2g.35gb:     0
  nvidia.com/mig-3g.71gb:     0
  pods:                       110

$ kubectl describe node <cluster-node> | grep -A20 "Allocatable:"
Allocatable:
  cpu:                        224
  memory:                     2113351248Ki
  nvidia.com/gpu:             0
  nvidia.com/mig-1g.18gb:     56
  nvidia.com/mig-2g.35gb:     0
  nvidia.com/mig-3g.71gb:     0
  pods:                       110
```

Both workers report `0` for the medium (`2g.35gb`) and large (`3g.71gb`) MIG
profiles, so those profiles are not provisioned anywhere on the cluster for
anyone. Each worker offers 56 of the smallest slice, which matches 8 physical
GPUs cut at 7 instances each.

## Account quota

```console
$ kubectl get resourcequota
NAME         AGE     REQUEST
user-quota   2d22h   requests.cpu: 8/8, requests.memory: 16Gi/16Gi,
                     requests.nvidia.com/gpu: 0/0,
                     requests.nvidia.com/mig-1g.18gb: 1/1,
                     requests.nvidia.com/mig-2g.35gb: 0/0,
                     requests.nvidia.com/mig-3g.71gb: 0/0

$ kubectl describe resourcequota
Name:                            user-quota
Namespace:                       <namespace>
Resource                         Used  Hard
--------                         ----  ----
limits.cpu                       8     8
limits.memory                    16Gi  16Gi
requests.cpu                     8     8
requests.memory                  16Gi  16Gi
requests.nvidia.com/gpu          0     0
requests.nvidia.com/mig-1g.18gb  1     1
requests.nvidia.com/mig-2g.35gb  0     0
requests.nvidia.com/mig-3g.71gb  0     0
```

A `ResourceQuota` called `user-quota` caps this namespace at exactly one
`1g.18gb` slice, and at zero of the two larger profiles. This is separate from
node capacity: the API server enforces it when the pod is submitted, before the
scheduler ever looks at the nodes.

## Asking for two slices in one pod

```console
$ kubectl apply -f pod-service-medium-test.yaml
Error from server (Forbidden): error when creating "pod-service-medium-test.yaml":
pods "mig-clubbed-test" is forbidden: exceeded quota: user-quota,
requested: limits.cpu=4,limits.memory=8Gi,requests.cpu=4,requests.memory=8Gi,
           requests.nvidia.com/mig-1g.18gb=2,
used:      limits.cpu=8,limits.memory=16Gi,requests.cpu=8,requests.memory=16Gi,
           requests.nvidia.com/mig-1g.18gb=1,
limited:   limits.cpu=8,limits.memory=16Gi,requests.cpu=8,requests.memory=16Gi,
           requests.nvidia.com/mig-1g.18gb=1
```

It came back `Forbidden` at admission rather than sitting in `Pending`, which
means the quota blocks it regardless of how many slices happen to be free on the
nodes.

## Storage

```console
$ df -hT $HOME
Filesystem     Type  Size  Used Avail Use% Mounted on
/dev/sda1      xfs    66T  1.2T   65T   2% /home

$ kubectl get pvc
No resources found in <namespace> namespace.

$ kubectl get storageclass
No resources found

$ cat /etc/exports
/home    <internal-ip>/16(rw,no_subtree_check,no_root_squash,async)
```

`/home` is a 66 TB local XFS disk on the head node, not a network mount, and it
is exported read-write to the cluster fabric `<internal-ip>/16`. The campus-facing
`<campus-subnet>` address is not in the export list.

There is no PVC and no StorageClass in the namespace, so we cannot create a
persistent volume ourselves. Pods reach the data by mounting `/home` from the
worker node, which relies on the workers mounting that export.

Docker is not available on the head node. Installing it asks for `snap` or
`apt`, and we do not have admin rights, so building our own image is not an
option from this account.

## Pod durability test

Before the allocation testing above, we created a standard pod from
`pod-service.yaml` and left it running to see whether resources held. It stayed
up cleanly for 41 hours with no restarts.

```console
$ kubectl get pods
NAME       READY   STATUS    RESTARTS   AGE
my-pod-1   1/1     Running   0          41h

$ kubectl delete pod my-pod-1
pod "my-pod-1" deleted
```

Pod creation and long-running pods work fine on this account. The limits below
are about allocation only.

## Summary

| Question | Answer | Where |
|---|---|---|
| Do medium or large slices exist on this cluster? | No, `0` allocatable on both workers | GPU capacity |
| Is the one-slice limit a per-account policy? | Yes, a hard `ResourceQuota` at `1/1` | Account quota |
| Can we ask for two small slices in one pod? | No, rejected at admission | Two slices in one pod |
| Would two slices behave as one bigger GPU? | No, MIG instances stay hardware-isolated | NVIDIA MIG User Guide |
| Can allocation change on a running pod? | No, fixed when the pod is created | Kubernetes Scheduling GPUs |
| Can we provision our own storage? | No PVC, no StorageClass | Storage |
| Can we build our own container image? | No, Docker is absent and we have no admin rights | Storage |
| Does pod creation work at all? | Yes, 41 hours with no restarts | Pod durability test |

Raising the allocation needs an administrator to edit the `user-quota` object in
this namespace. Anything above `1g.18gb` additionally needs the MIG geometry on
the workers to be reconfigured. Neither is reachable from a pod spec.

## Why we are asking for more than one slice

We are not asking for a bigger slice. At the image sizes we work with, 18 GB of
GPU memory is not the constraint, and a larger profile would not speed anything
up.

The constraint is that we can only hold one slice at a time, so everything runs
back to back. Our evaluation protocol requires three independent runs of the
same configuration with different random seeds, because a single run does not
show how much of a result is real and how much is run-to-run noise. Those three
runs are completely independent of each other and need no communication between
GPUs, so with three slices they would finish in roughly the time one takes now.
With one slice they take three times as long, and every comparison we run
afterwards inherits that delay.

Three concurrent `1g.18gb` slices would therefore cut our turnaround by about
two thirds at the same total GPU cost, since we would be using the same number
of slice-hours either way.
