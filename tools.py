import webbrowser
import os
import json
import numpy as np
from jinja2 import Template

'''Tools for io, dataset handling, browser based 3D visualization, and numpy array to 3D mesh conversion'''

def get_path(dirname, fname=''):
    '''returns absolute path for file in a directory'''
    #only works when called from same dir
    abspath = os.path.dirname(__file__)
    dirpath = os.path.join(abspath, dirname)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return os.path.join(dirpath, fname)

def generate_coords(size):
    '''input coords for nn, vector of [x, y, radius]'''
    x = np.arange(0, size)
    y = np.arange(0, size)
    z = np.arange(0, size)
    x_points, y_points, z_points = np.meshgrid(x, y, z)
    center = size / 2
    tmp = []
    for i in range(x_points.shape[0]):
        for j in range(y_points.shape[1]):
            for k in range(z_points.shape[2]):
                dx = x_points[i, j, k]
                dy = y_points[i, j, k] 
                dz = z_points[i, j, k] 
                tmp.append([dx , dy , dz , np.sqrt((dx - center) ** 2 + (dy - center) ** 2 + (dz - center) ** 2)])

    data = np.array(tmp, dtype=np.float32)
    data_min, data_max = np.amin(data), np.amax(data)
    normal = (data - data_min)  / (data_max - data_min) 
    return normal

def gen_coord_datasets():
    '''Generate and save coordinate datasets for disk access.
       Coordinate datasets are in powers of 2 sizes 8:256'''

    if not os.path.exists('coord_datasets'):
        for i in range(3, 9, 1):
            size = 2 ** i
            save_path = get_path('coord_datasets', 'data{}'.format(size))
            data = generate_coords(size)
            np.save(save_path, data)
        print('datasets generated')
    else:
        print('datasets already generated')

def load_coord_dataset(size):
    data_path = get_path('coord_datasets', 'data{}.npy'.format(size))
    return np.load(data_path)


def np2vox(bin_array):
    '''Convert binary numpy ndarray to indexed 3D mesh data'''
    
    def scale_verts(x, y, z, face_type):
        x *= 2
        y *= 2
        z *= 2
        verts = []

        if face_type == 0: 
            verts = [(0.0 + x, 2.0 + y, 2.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 1: 
            verts = [(0.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z),               
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 2: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z)]

        elif face_type == 3: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z)]

        elif face_type == 4: 
            verts = [(2.0 + x, 2.0 + y, 0.0 + z),
                     (2.0 + x, 0.0 + y, 0.0 + z),
                     (2.0 + x, 2.0 + y, 2.0 + z),
                     (2.0 + x, 0.0 + y, 2.0 + z)]

        elif face_type == 5: 
            verts = [(0.0 + x, 2.0 + y, 0.0 + z),
                     (0.0 + x, 0.0 + y, 0.0 + z),
                     (0.0 + x, 2.0 + y, 2.0 + z),
                     (0.0 + x, 0.0 + y, 2.0 + z)]
        
        return verts

    def scale_faces(scale, face_type):

        faces = np.array([[[0, 1, 3, 2]],
                          [[0, 1, 3, 2]],
                          [[1, 0, 2, 3]],
                          [[1, 0, 2, 3]],
                          [[1, 0, 2, 3]],
                          [[0, 1, 3, 2]]])
  
        scaled_faces = faces[face_type] + 4 * scale
        return scaled_faces.tolist()
        
    print('--> BUILDING MESH')
    print('--> VOXEL VOLUME:', np.count_nonzero(bin_array))
    bin_array = np.fliplr(np.flipud(bin_array))
    verts = []
    faces = []
    x, y, z = bin_array.shape[0], bin_array.shape[1], bin_array.shape[2]
    x_max, y_max, z_max = x - 1, y - 1, z - 1
    count = 0
    bin_array[0, :, :] = 0
    bin_array[:, 0, :] = 0
    bin_array[:, :, 0] = 0
    bin_array[x_max, :, :] = 0
    bin_array[:, y_max, :] = 0
    bin_array[:, :, z_max] = 0
        
    for i in range(x): 
        for j in range(y):
            for k in range(z):
                current = bin_array[i,j,k]
                if i == x_max:
                    i = x_max - 1
                elif i == 0:
                    i = 1
                if j == y_max:
                    j = y_max - 1
                elif j == 0:
                    j = 1
                if k == z_max:
                    k = z_max - 1 
                elif k == 0:
                    k = 1

                surrounding = np.array([bin_array[i,j+1,k],    
                                        bin_array[i-1,j,k], 
                                        bin_array[i,j-1,k], 
                                        bin_array[i+1,j,k], 
                                        bin_array[i,j,k+1], 
                                        bin_array[i,j,k-1]], dtype=int)
                if current == 1:
                    for num in range(len(surrounding)):
                        if surrounding[num] == 0:
                            verts += scale_verts(k, i, j, num)
                            faces += scale_faces(count, num)
                            count += 1
    return verts, faces

def render_voxels(voxels):
    '''Render np array in the browser as a mesh using np2vox func and three.js lib'''
    verts, faces = np2vox(voxels)
    mesh_data = {'verts': verts, 'faces': faces}
    json_mesh = json.dumps(mesh_data)

    html = Template('''<html>
            <head>
                <title>Viewer</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
                <link rel="stylesheet" type="text/css" href="css/styles1.css"/>
            </head>
            <body>
                <canvas id="canvas"></canvas>
                <div id="top_panel"></div>
                <div id="bottom_panel">  
                </div>

                <script src="js/three.min.js"></script>
                <script src="js/OrbitControls.js"></script>
              
                <script>
                    
                    var renderer = new THREE.WebGLRenderer({canvas: document.getElementById('canvas'), antialias: true});
                  
                    var camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 10000);
                    var scene = new THREE.Scene();

                    //renderer
                    renderer.setClearColor(0x37373B); 
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    document.body.appendChild(renderer.domElement);
                    
                    //scene and camera setup
                    camera.position.set(-200, 200, -200);
                    camera.up = new THREE.Vector3(0, 1, 0);
                    camera.lookAt(new THREE.Vector3(0, 0, 0))
                    scene.add(camera);
                
                    //controls
                    var controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.target.set( 0, 0, 0);
                    controls.update();
                    //controls.minDistance = 150;
                    controls.maxDistance = 2000;
                    controls.zoomSpeed = 0.5;
                    controls.enablePan = true;
                    controls.rotateSpeed = 0.5;

                    //lights
                    var light1 = new THREE.AmbientLight(0xffffff, 1.0);
                    scene.add(light1);

                    var light2 = new THREE.PointLight(0xff3333, 0.75);
                    light2.position.set(100, 200, 500);
                    scene.add(light2)
                    var light4 = new THREE.PointLight(0x3333ff, 0.75);
                    light2.position.set(-100, 200, -500);
                    scene.add(light4)

                    var light3 = new THREE.SpotLight(0xddddff, 1);
                    light3.position.set(-300, -300, 300);
                    scene.add(light3);

                    window.addEventListener('resize',windowResize, false);

                    function windowResize(){
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize( window.innerWidth, window.innerHeight);
                    }

                    function render() {
                        requestAnimationFrame(render);
                        renderer.render(scene, camera);
                    }

                    var mesh_data = JSON.parse(JSON.stringify({{ data }}));
                    console.log(mesh_data)
                    var verts = mesh_data.verts;
                    var faces = mesh_data.faces;
                
                    var geometry = new THREE.Geometry();

                    for (i=0; i < verts.length; i++){
                        geometry.vertices.push(new THREE.Vector3(verts[i][1], verts[i][2], verts[i][0]));
                    }
                    for (i=0; i < faces.length; i++){
                        geometry.faces.push(new THREE.Face3(faces[i][0], faces[i][1], faces[i][2]));
                        geometry.faces.push(new THREE.Face3(faces[i][2], faces[i][3], faces[i][0]));
                    }
                    console.log('computing normals');
                    geometry.computeBoundingSphere();
                    geometry.computeFaceNormals();
                    geometry.computeVertexNormals();
                    console.log('building scene');

                    var material = new THREE.MeshLambertMaterial({

                                color: 0x616c72,
                            
                                
                    });
                    var material1 = new THREE.MeshLambertMaterial({

                                color: 0x000000,
                                wireframe: true,
                                transparent: true,
                                opacity: 0.7,                                
                    });
                    
                    
                    var voxmesh = new THREE.Mesh(geometry, material);
                    var voxmeshW = new THREE.Mesh(geometry, material1);
                    var geo = new THREE.PlaneGeometry(200, 200);
                    var mat = new THREE.MeshLambertMaterial({

                                color: 0xe0e3e5,
                                wireframe: false,
                                transparent: true,
                                opacity: 0.7,
                                side: THREE.DoubleSide
                                
                    });
                    var plane = new THREE.Mesh(geo, mat);
                    plane.rotateX(Math.PI / 2);
                    scene.add(plane);
                    scene.add(voxmesh);
                   // scene.add(voxmeshW);
                    var x = {{x}};
                    var y = {{y}};
                    var z = {{z}};
                    voxmesh.position.set(x, y, z);
                    //voxmeshW.position.set(x, y, z);

                    var cube_geo = new THREE.BoxGeometry(5, 5, 5);
                    var cube_mat = new THREE.MeshLambertMaterial({ color: 0x442222});
                    var cube = new THREE.Mesh(cube_geo, cube_mat);
                    scene.add(cube);
                    cube.position.set(-102.5, 0, -102.5);

                    render();
                </script>
            </body>
        </html>''')
    
    new_html = html.render(data=json_mesh, x=-voxels.shape[0], y=voxels.shape[1] / 2, z=-voxels.shape[2])
    path = get_path('templates', 'template.html')
    with open(path, 'w') as f:
        f.write(new_html)
    webbrowser.open(path, new=2)

def render_voxel_ani(vox_list):
    '''Render latent space traversal as animation.
       Displays np array in the browser as a mesh using np2vox func and three.js lib'''    
    verts_list = []
    faces_list = []
    for vox in vox_list:
        # compute the mesh data for each voxel array
        verts, faces = np2vox(vox)
        verts_list.append(verts)
        faces_list.append(faces)
    # json version of list of mesh data
    mesh_data = {'verts': verts_list, 'faces': faces_list}
    json_meshes = json.dumps(mesh_data)
    html = Template('''<html>
            <head>
                <title>Viewer</title>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, user-scalable=no, minimum-scale=1.0, maximum-scale=1.0">
                <link rel="stylesheet" type="text/css" href="css/styles1.css"/>
            </head>
            <body>
                <canvas id="canvas"></canvas>
                <div id="top_panel"></div>
                <div id="bottom_panel">  
                </div>

                <script src="js/three.min.js"></script>
                <script src="js/OrbitControls.js"></script>
              
                <script>
                    var renderer = new THREE.WebGLRenderer({canvas: document.getElementById('canvas'), antialias: true});
                    var camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 10000);
                    var scene = new THREE.Scene();

                    renderer.setClearColor(0x37373B); 
                    renderer.setSize(window.innerWidth, window.innerHeight);
                    renderer.setPixelRatio(window.devicePixelRatio);
                    document.body.appendChild(renderer.domElement);
                    
                    //scene and camera setup
                    camera.position.set(-200, 200, -200);
                    camera.up = new THREE.Vector3(0, 1, 0);
                    camera.lookAt(new THREE.Vector3(0, 0, 0))
                    scene.add(camera);
                
                    //controls
                    var controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.target.set( 0, 0, 0);
                    controls.update();
                    //controls.minDistance = 150;
                    controls.maxDistance = 2000;
                    controls.zoomSpeed = 0.5;
                    controls.enablePan = true;
                    controls.rotateSpeed = 0.5;

                    //lights
                    var light1 = new THREE.AmbientLight(0xffffff, 1.0);
                    scene.add(light1);
                    var light2 = new THREE.PointLight(0xff3333, 0.75);
                    light2.position.set(100, 200, 500);
                    scene.add(light2)
                    var light4 = new THREE.PointLight(0x3333ff, 0.75);
                    light2.position.set(-100, 200, -500);
                    scene.add(light4)
                    var light3 = new THREE.SpotLight(0xddddff, 1);
                    light3.position.set(-300, -300, 300);
                    scene.add(light3);

                    window.addEventListener('resize',windowResize, false);

                    function windowResize(){
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize( window.innerWidth, window.innerHeight);
                    }

                    function render() {
                        requestAnimationFrame(render);
                        renderer.render(scene, camera);
                    }

                    var material = new THREE.MeshLambertMaterial({color: 0x616c72});
                    var material1 = new THREE.MeshLambertMaterial({
                                color: 0x000000,
                                wireframe: true,
                                transparent: true,
                                opacity: 0.7,                                
                    });

                    var geo = new THREE.PlaneGeometry(200, 200);
                    var mat = new THREE.MeshLambertMaterial({
                                color: 0xe0e3e5,
                                wireframe: false,
                                transparent: true,
                                opacity: 0.7,
                                side: THREE.DoubleSide 
                    });

                    var plane = new THREE.Mesh(geo, mat);
                    plane.rotateX(Math.PI / 2);
                    scene.add(plane);
                    var x = {{x}};
                    var y = {{y}};
                    var z = {{z}};
                    var cube_geo = new THREE.BoxGeometry(5, 5, 5);
                    var cube_mat = new THREE.MeshLambertMaterial({ color: 0x442222});
                    var cube = new THREE.Mesh(cube_geo, cube_mat);
                    scene.add(cube);
                    cube.position.set(-102.5, 0, -102.5);
                    
                    // mesh
                    var mesh_data = JSON.parse(JSON.stringify({{ mesh_list }}));
                    var verts_list = mesh_data.verts;
                    var faces_list = mesh_data.faces;

                    for (j=0; j < verts_list.length; j++)
                    {
                        // build each object as a three js mesh
                        verts = verts_list[j];
                        faces = faces_list[j];
                        
                        var geometry = new THREE.Geometry();
                        for (i=0; i < verts.length; i++){
                            geometry.vertices.push(new THREE.Vector3(verts[i][1], verts[i][2], verts[i][0]));
                        }
                        for (i=0; i < faces.length; i++){
                            geometry.faces.push(new THREE.Face3(faces[i][0], faces[i][1], faces[i][2]));
                            geometry.faces.push(new THREE.Face3(faces[i][2], faces[i][3], faces[i][0]));
                        }
                        geometry.computeBoundingSphere();
                        geometry.computeFaceNormals();
                        geometry.computeVertexNormals();
                        var voxmesh = new THREE.Mesh(geometry, material);
                        voxmesh.name = 'mesh' + i.toString();
                        voxmesh.visible = false;
                        voxmesh.position.set(x, y, z);
                        scene.add(voxmesh);
                        console.log('built mesh');
                    }

                    var delay = {{ speed }};
                    var i = 7;
                    setInterval(function()
                    {
                        if (i == scene.children.length - 1){
                            scene.children[i-1].visible = false;
                            i = 7;
                        } else {
                            if (i != 7){
                                scene.children[i-1].visible = false;
                            }
                            scene.children[i].visible = true;
                            i++;
                        }
                    }, delay);

                    render();
                </script>
            </body>
        </html>''')

    # animation speed line eq
    x = [5, 20]
    y = [75, 8]
    c = np.polyfit(x, y, 1)
    line = np.poly1d(c)
    new_html = html.render(mesh_list=json_meshes, 
                            x=-vox_list[0].shape[0], 
                            y=vox_list[0].shape[1] / 2, 
                            z=-vox_list[0].shape[2],
                            speed=line(len(vox_list)) * len(vox_list))
    
    path = get_path('templates', 'template_ani.html')
    with open(path, 'w') as f:
        f.write(new_html)
    webbrowser.open(path, new=2) 