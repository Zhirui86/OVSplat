for d in ../datasets/scans/* ; 
do     
    scene=$(echo $d | cut -d'/' -f4)
    echo "Processing scene: $scene"
    python reader.py --scene $scene 
done


# ls scans/* | cut -d'/' -f2 | xargs -P 16 -I {} sh -c 'echo "Processing scene: {}"; python reader.py --scene {}'
