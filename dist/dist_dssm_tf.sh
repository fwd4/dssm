#! /bin/bash

declare -a HOSTS=("tf20-1" "tf21-1" "tf20-2" "tf21-2" "tf20-3" "tf21-3" "tf20-4" "tf21-4")
declare -a IPS=("10.1.0.200" "10.1.0.204" "10.1.0.201" "10.1.0.205" "10.1.0.202" "10.1.0.206" "10.1.0.203" "10.1.0.207" )
PORT=2222
USER=tensorflow-20

# for HOSTNAME in ${HOSTS[@]}; do
# 	sshpass -p 123 ssh -l ${USER} ${HOSTNAME} "echo 123 | sudo -S reboot"
# done

# sleep 20


# Default Setting
N_HOSTS=8
N_WORKERS=4
N_PS=2
HALT=0

WORK_DIR=git/tensorflow/tensorflow/models/dssm

for i in "$@"; do
	case $i in 
		-np=*|--num_ps=*)
		N_PS="${i#*=}"
		;;
		-nw=*|--num_worker=*)
		N_WORKERS="${i#*=}"
		;;
		--kill)
		HALT=1
		;;
		*)
		;;
	esac
done
	

if (( $((${N_WORKERS} + ${N_PS})) > ${N_HOSTS} )); then
	echo "Total #workers and #parameter servers exceeds available #machines: ${N_HOSTS}"
	exit
fi


PSS=(${HOSTS[@]:0:${N_PS}})
WORKERS=(${HOSTS[@]:${N_PS}:${N_WORKERS}})


PS_HOSTS=""
for ((i = 0; i < N_PS - 1; ++i)) ; do
	PS_HOSTS="${PS_HOSTS}${IPS[$i]}:${PORT},"
done
PS_HOSTS="${PS_HOSTS}${IPS[$i]}:${PORT}"
echo "ps_hosts=${PS_HOSTS}"

WORKER_HOSTS=""
for ((i = N_PS; i < N_PS + N_WORKERS - 1; ++i)) ; do
	WORKER_HOSTS="${WORKER_HOSTS}${IPS[$i]}:${PORT},"
done
WORKER_HOSTS="${WORKER_HOSTS}${IPS[$i]}:${PORT}"
echo "worker_hosts=${WORKER_HOSTS}"

echo ${PSS[@]}
echo ${WORKERS[@]}

# Kill All Existing Tasks
for HOSTNAME in ${HOSTS[@]} ; do
	sshpass -p 123 ssh -l ${USER} ${HOSTNAME} "echo 123 | sudo -S fuser -k 2222/tcp"
	sshpass -p 123 ssh -l ${USER} ${HOSTNAME} "rm -rf /tmp/dssm-dist"
	# sshpass -p 123 ssh -l ${USER} ${HOSTNAME} "echo 123 | sudo -S apt-get update"
	# sshpass -p 123 ssh -l ${USER} ${HOSTNAME} "echo 123 | sudo -S apt-get -y upgrade"
	sshpass -p 123 scp ~/${WORK_DIR}/dist/sync-dssm-dist.py ${USER}@${HOSTNAME}:~/${WORK_DIR}
	echo "$HOSTNAME ready"
done

if (($HALT)); then
	exit
fi	

# Start Parameter Servers
temp=0
for HOSTNAME in ${PSS[@]} ; do
	#echo "Starting parameter server on ${HOSTNAME}"
	sshpass -p 123 ssh -l ${USER} ${HOSTNAME} \
			"cd git/tensorflow/tensorflow/models/dssm && \
			python sync-dssm-dist.py --ps_hosts='${PS_HOSTS}' \
			--worker_hosts='${WORKER_HOSTS}' \
			--num_workers=${N_WORKERS} --job_name=ps --task_index=${temp}" &
	echo "Started parameter server ${temp} on ${HOSTNAME}"
	temp=$((temp+1))
done

# Start Workers
temp=0
for HOSTNAME in ${WORKERS[@]} ; do
	# echo "python sync-dssm-dist.py --ps_hosts='${PS_HOSTS}' \
	# 		--worker_hosts='${WORKER_HOSTS}' \
	# 		--num_workers=${N_WORKERS} --job_name=worker --task_index=${temp}" 
	sshpass -p 123 ssh -l ${USER} ${HOSTNAME} \
			"cd git/tensorflow/tensorflow/models/dssm && \
			python sync-dssm-dist.py --ps_hosts='${PS_HOSTS}' \
			--worker_hosts='${WORKER_HOSTS}' \
			--num_workers=${N_WORKERS} --job_name=worker --task_index=${temp} > run.log" &
	echo "Starting worker ${temp} on ${HOSTNAME}"
	temp=$((temp+1))
done
