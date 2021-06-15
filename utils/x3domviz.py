import os

def generate_html_viz(datab_object):
    datab_object.cursor.execute('SELECT ST_AsX3D(bbox), ST_AsX3D(cbb), ST_AsX3D(tophsproj),'
                                'ST_AsX3D(bottomhsproj),ST_AsX3D(lefthsproj),ST_AsX3D(righthsproj),'
                                'ST_AsX3D(fronthsproj),ST_AsX3D(backhsproj)  '
                                'FROM single_snap;')
    tgtp= os.path.join(os.environ['HOME'], 'hsdump.html')
    with open(tgtp, 'w') as outd:
        # Write html heading
        outd.write(
            '<!DOCTYPE html><html><head><meta encoding="utf-8"><script src="http://www.x3dom.org/download/dev/x3dom.js"></script><link rel="stylesheet" '
            'href="http://www.x3dom.org/download/dev/x3dom.css"></head><body>\
                    <x3d xmlns="http://www.x3dom.org/x3dom" showStat="false" showLog="false" x="0px" y="0px" width="1920px" height="1080px"><scene>\
                    <viewpoint position="0 0 10"></viewpoint>' + '\n')
        for i,r in enumerate(datab_object.cursor.fetchall()):
            # bbox is black (default)
            outd.write('<shape><appearance><material></material></appearance>' + '\n')
            outd.write(r[0].replace('FaceSet','LineSet') + '\n')
            outd.write('</shape>' + '\n')
            if i==0: #draw all boxes/hs only for one for readability
                # CBBs are blue
                outd.write('<shape><appearance><material emissiveColor="0. 0. 1.0"></material></appearance>' + '\n')
                outd.write(r[1].replace('FaceSet','LineSet') + '\n')
                outd.write('</shape>' + '\n')
                #tophs are red
                #outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                #outd.write(r[2].replace('FaceSet','LineSet') + '\n')
                #outd.write('</shape>' + '\n')
                #btmhs green
                #outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                #outd.write(r[3].replace('FaceSet', 'LineSet') + '\n')
                #outd.write('</shape>' + '\n')
                # lefths are red
                # outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                # outd.write(r[4].replace('FaceSet', 'LineSet') + '\n')
                # outd.write('</shape>' + '\n')
                # righths green
                # outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                # outd.write(r[5].replace('FaceSet', 'LineSet') + '\n')
                # outd.write('</shape>' + '\n')
                # fronths are red
                outd.write('<shape><appearance><material emissiveColor="1.0 0. 0.0"></material></appearance>' + '\n')
                outd.write(r[6].replace('FaceSet', 'LineSet') + '\n')
                outd.write('</shape>' + '\n')
                # backhs green
                outd.write('<shape><appearance><material emissiveColor="0. 1.0 0."></material></appearance>' + '\n')
                outd.write(r[7].replace('FaceSet', 'LineSet') + '\n')
                outd.write('</shape>' + '\n')
        # closing nodes
        outd.write('<shape><plane></plane></shape></scene></x3d></body></html>')
    print("HTML report of 3D boxes created under %s" % tgtp)