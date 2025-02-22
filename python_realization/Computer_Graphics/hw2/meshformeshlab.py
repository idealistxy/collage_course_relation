import numpy as np


# 读取.obj文件
def load_obj(file_path):
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith('f '):
                faces.append([int(v.split('/')[0]) - 1 for v in line.split()[1:]])

    return np.array(vertices), np.array(faces)


# 进行三维网格细分
def loop_subdivision(vertices, faces, num_subdivisions=1):
    for _ in range(num_subdivisions):
        #继承先前工作
        new_vertices = vertices.copy()
        new_faces = faces.copy()
        #遍历每个面开始细分
        for face in faces:
            v1, v2, v3 = [vertices[i] for i in face]
            #取三边中点
            midpoints = [(v1 + v2) / 2, (v2 + v3) / 2, (v3 + v1) / 2]
            #添加中间点到点元组
            new_vertices = np.vstack((new_vertices, midpoints))
            #得到中间点索引
            mid1, mid2, mid3 = len(new_vertices) - 3, len(new_vertices) - 2, len(new_vertices) - 1
            #将点组合成边添加到面元组
            new_faces = np.vstack((new_faces, [[face[0], mid1, mid3], [mid1, face[1], mid2], [mid3, mid2, face[2]], [mid1, mid2, mid3]]))

        vertices = new_vertices
        faces = new_faces
    #返回处理后的点集和面集
    return vertices, faces


# 保存.obj文件
def save_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# 文件路径
file_path = 'D:/大三课程/计算机图形学/HW2/HW2_datasets/缃戞牸/Horse.obj'

# 读取.obj文件
prevertices, prefaces = load_obj(file_path)

# 进行三维网格细分
nowvertices, nowfaces = loop_subdivision(prevertices, prefaces, num_subdivisions=1)

# 将面由数组转换为集合类型(将原本的面剔除)
set_prefaces = set(map(tuple, prefaces))
set_nowfaces = set(map(tuple, nowfaces))
# 计算差集
new_faces_set = set_nowfaces - set_prefaces
# 将差集转换回列表
newfaces = [list(face) for face in new_faces_set]
# 保存细分后的.obj文件
save_obj('D:/大三课程/计算机图形学/HW2/HW2_datasets/缃戞牸/Horseafterdealing.obj', nowvertices, newfaces)
