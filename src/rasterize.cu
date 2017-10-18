/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define BILINEAR 0
#define PERSPECTIVE 1


#define SSAA 4
namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec3 col;
		
		 VertexAttributeTexcoord texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int diffuseTexWidth;
		 int diffuseTexHeight;
		// int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
//The image size we want show
static int imageWidth = 0;
static int imageHeight = 0;

static int totalNumPrimitives = 0;
static int curPrimitiveBeginId = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;
static bool* dev_flag = NULL;
cudaEvent_t start, stop;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */

__host__ __device__ static
void Bresenham(glm::vec3 v1, glm::vec3 v2, Fragment* fragment, const int width, const int height) {
	float x1 = v1[0], x2 = v2[0], y1 = v1[1], y2 = v2[1];

	glm::clamp(x1, 0.f, float(width - 1));
	glm::clamp(x2, 0.f, float(width - 1));
	glm::clamp(y1, 0.f, float(height - 1));
	glm::clamp(y2, 0.f, float(height - 1));

	glm::vec3 linecolor(0.0, 0.8, 1.f);
	int pixelIdx = int(x1 + 0.5) + int(y1 + 0.5)*width;
	fragment[pixelIdx].color = linecolor;

	float deltax = fabs(x2 - x1), deltay = fabs(y2 - y1);
	if (int(deltax + 0.5) == 0 && int(deltay + 0.5) == 0)
		return;
	pixelIdx = int(x2 + 0.5) + int(y2 + 0.5)*width;
	fragment[pixelIdx].color = linecolor;
	bool flag = false;
	if (deltax<deltay)//Always keep the slope in the range of (0,1)
	{
		flag = true;
		swap(x1, y1);
		swap(x2, y2);
		swap(deltax, deltay);
	}
	float x_coord = x1, y_coord = y1, dX = (x2 - x1)>0 ? 1 : -1, dY = (y2 - y1)>0 ? 1 : -1, dT = 2 * (deltay - deltax), \
		dS = 2 * deltay, d = 2 * deltay - deltax;
	//x_coord and y_coord are the coordinates of the point we want to colour.dX and dY are the increment of
	//x_coord and ycoord in each loop. dS and dT are the y distance between the intersection point scan line /
	//(y and y+1)
	while (abs(int(x_coord - x2)) != 0) {
		if (d<0)
			d += dS;
		else
		{
			d += dT;
			y_coord += dY;
		}
		float scale = fabs((x_coord - x1) / deltax);
		if (flag)
			pixelIdx = int(y_coord + 0.5) + int(x_coord + 0.5)*width;
		else
			pixelIdx = int(x_coord + 0.5) + int(y_coord + 0.5)*width;
		fragment[pixelIdx].color = linecolor;
		x_coord += dX;
	}
}
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image,int ssaa) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
		for(int i=0;i<ssaa;i++)
			for (int j = 0; j < ssaa; j++) {
				color.x += glm::clamp(image[x*ssaa + i + (y*ssaa + j)*w*ssaa].x, 0.f, 1.f)*255.f;
				color.y += glm::clamp(image[x*ssaa + i + (y*ssaa + j)*w*ssaa].y, 0.f, 1.f)*255.f;
				color.z += glm::clamp(image[x*ssaa + i + (y*ssaa + j)*w*ssaa].z, 0.f, 1.f)*255.f;
			}
		color /= (float)(ssaa*ssaa);
        //color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        //color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        //color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer,int renderMode) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);
	auto &curFrag = fragmentBuffer[index];
	float texWidth = curFrag.diffuseTexWidth;
	float texHeight = curFrag.diffuseTexHeight;
	glm::vec3 curEyePos = curFrag.eyePos;
	glm::vec3 curEyeNor = curFrag.eyeNor;
	glm::vec3 lightPos(60.f, 60.f, 60.f);
	glm::vec3 lightEmittance(1.f, 1.f, 1.f); 
	glm::vec3 lightDir = glm::normalize((lightPos - curEyePos));
	glm::vec3 EyeDir = glm::normalize(-curEyePos);	

	//Phong reflection Model L=kdD + I(ksS + kaA)
	float ka = 0.1f;//ambient coefficient
	float kd = glm::max(0.f, glm::dot(curEyeNor, lightDir));//diffuse coefficient
	float ks = 0;//
	float Shiness = 20.f;

	glm::vec3 ambientColor(1.f);
	glm::vec3 diffuseColor;
	glm::vec3 specularColor(1.f);
	float cosValue = glm::max(0.f, glm::dot(curEyeNor, lightDir));
	
	glm::vec3 renderColor = glm::vec3(0.f);
	framebuffer[index] = glm::vec3(0.f);
	if (renderMode == 0) {
		if (x < w && y < h) {
			// TODO: add your fragment shader code here
			float uFloat = curFrag.texcoord0.x * texWidth;
			float vFloat = curFrag.texcoord0.y* texHeight;
			int u = (int)uFloat;
			int v = (int)vFloat;
			int uvIdx = 3 * (u + v * texWidth);
			TextureData* texture = curFrag.dev_diffuseTex;
#if BILINEAR
			if (texture) {
				int leftdown = 3 * (u + v * texWidth);
				int rightdown = leftdown + 3;
				int leftup = leftdown + 3 * texWidth;
				int rightup = leftup + 3;
				glm::vec3 textureLD(texture[leftdown] / 255.f, texture[leftdown + 1] / 255.f, texture[leftdown + 2] / 255.f);
				glm::vec3 textureRD(texture[rightdown] / 255.f, texture[rightdown + 1] / 255.f, texture[rightdown + 2] / 255.f);
				glm::vec3 textureLU(texture[leftup] / 255.f, texture[leftup + 1] / 255.f, texture[leftup + 2] / 255.f);
				glm::vec3 textureRU(texture[rightup] / 255.f, texture[rightup + 1] / 255.f, texture[rightup + 2] / 255.f);
				glm::vec2 coord(uFloat - u, vFloat - v);
				diffuseColor = BilinearInterpolation(coord, textureLD, textureRD, textureLU, textureRU);
			}
			else diffuseColor = curFrag.color;
#else
			if (texture)
				diffuseColor = glm::vec3(texture[uvIdx] / 255.f, texture[uvIdx + 1] / 255.f, texture[uvIdx + 2] / 255.f);
			else diffuseColor = curFrag.color;
#endif

			//Lambert
			//renderColor = diffuseColor * cosValue;


			//Blinning Phong
			glm::vec3 H = glm::normalize(lightDir + EyeDir);
			ks = 0.75 * pow(glm::max(0.f, glm::dot(H, curEyeNor)), Shiness);
			renderColor = ambientColor * ka * lightEmittance + lightEmittance * (kd * cosValue * diffuseColor + ks * specularColor);
			//renderColor= curFrag.color;
			framebuffer[index] = renderColor;
		}
	}
	else 
		framebuffer[index] = curFrag.color;
	//framebuffer[index] = glm::vec3(1.f, 0.f, 0.f);
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
	
	width = SSAA*w;
    height = SSAA*h;
	imageWidth = w;
	imageHeight = h;

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));
	


	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
		cudaMalloc(&dev_flag, totalNumPrimitives * sizeof(bool));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space
		glm::vec4 pos= glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 vPos = MVP*pos;
		vPos/= vPos.w;
		vPos.x= 0.5f * (float)width * (1.f - vPos.x);
		vPos.y= 0.5f * (float)height * (1.f - vPos.y);
		primitive.dev_verticesOut[vid].pos = vPos;
		primitive.dev_verticesOut[vid].eyePos= glm::vec3(MV * pos);
		primitive.dev_verticesOut[vid].eyeNor= glm::normalize(MV_normal * primitive.dev_normal[vid]);
		primitive.dev_verticesOut[vid].diffuseTexWidth = primitive.diffuseTexWidth;
		primitive.dev_verticesOut[vid].diffuseTexHeight = primitive.diffuseTexHeight;
		if (primitive.dev_texcoord0 ) {
			primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		}
		if (primitive.dev_diffuseTex)
		{
			primitive.dev_verticesOut[vid].dev_diffuseTex = primitive.dev_diffuseTex;
		}

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array

		
	}
}

__global__
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType].col = glm::vec3(1.0f, 1.0f, 1.0f);
		}
		//int pid = iid / 3;
		//dev_primitives[pid + curPrimitiveBeginId].v[iid % 3] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
		//dev_primitives[pid + curPrimitiveBeginId].v[iid % 3].col = glm::vec3(1.0f, 1.0f, 1.0f);


		// TODO: other primitive types (point, line)
	}

}
__global__ void backFaceCulling(int numPrimitives, Primitive* primitives, bool *flag)
{
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	Primitive targetPrimitive = primitives[pid];
	if (pid < numPrimitives)
	{
		glm::vec3 l1 = targetPrimitive.v[1].eyePos - targetPrimitive.v[0].eyePos;
		glm::vec3 l2 = targetPrimitive.v[2].eyePos - targetPrimitive.v[0].eyePos;
		glm::vec3 nor = glm::cross(l1, l2);
		flag[pid] = nor.z < 0.f ? false : true;
	}

}
void CompressPrimitives(int &numPrimitives, Primitive* primitives, bool *flag)
{
	thrust::device_ptr<bool> dev_ptrFlag(flag);
	thrust::device_ptr<Primitive> dev_primitives(primitives);
	thrust::remove_if(dev_primitives, dev_primitives + numPrimitives, dev_ptrFlag, thrust::logical_not<bool>());
	numPrimitives = thrust::count_if(dev_ptrFlag, dev_ptrFlag + numPrimitives, thrust::identity<bool>());
}

__global__ void kernRasterize(Fragment* fragment,int* depth, Primitive* primitive, const int numPrimitives,const int width, const int height,int renderMode) {
	int pid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (pid < numPrimitives) {
		Primitive targetPrimitive = primitive[pid];
		glm::vec3 tri[3] = { glm::vec3(targetPrimitive.v[0].pos),
			glm::vec3(targetPrimitive.v[1].pos),
			glm::vec3(targetPrimitive.v[2].pos) };
		glm::vec3 triEyePos[3] = { glm::vec3(targetPrimitive.v[0].eyePos),
			glm::vec3(targetPrimitive.v[1].eyePos),
			glm::vec3(targetPrimitive.v[2].eyePos) };
		glm::vec3 triEyeNor[3] = { targetPrimitive.v[0].eyeNor,
			targetPrimitive.v[1].eyeNor,
			targetPrimitive.v[2].eyeNor };
		glm::vec2 Texcoord[3] = { targetPrimitive.v[0].texcoord0,
			targetPrimitive.v[1].texcoord0,
			targetPrimitive.v[2].texcoord0 };
		int pixelIdx = 0;
		if (renderMode == 0) {
			AABB targetBound = getAABBForTriangle(tri);
			//int iMaxx = (-targetBound.min.x + 1.0f) * 0.5f * float(width);
			//int iMinx = (-targetBound.max.x + 1.0f) * 0.5f * float(width);
			//int iMaxy = (-targetBound.min.y + 1.0f) * 0.5f * float(height);
			//int iMiny = (-targetBound.max.y + 1.0f) * 0.5f * float(height);

			int iMinx = glm::max((int)targetBound.min.x, 0);
			int iMaxx = glm::min((int)targetBound.max.x, width - 1);
			int iMiny = glm::max((int)targetBound.min.y, 0);
			int iMaxy = glm::min((int)targetBound.max.y, height - 1);
			glm::vec2 curPixelCord;
			glm::vec3 curBaryCord;//project the 2d pixel coordinate into 3d space.
			int curPixelDepth = 1; // Store the depth of the current pixel.
			for (int x = iMinx; x <= iMaxx; x++) {
				for (int y = iMiny; y <= iMaxy; y++) {
					curPixelCord.x = x;
					curPixelCord.y = y;
					pixelIdx = y * width + x;
					auto& curFrag = fragment[pixelIdx];
					curBaryCord = calculateBarycentricCoordinate(tri, curPixelCord);
					if (isBarycentricCoordInBounds(curBaryCord)) {

						float tempDepth = -getZAtCoordinate(curBaryCord, tri);
						glm::clamp(tempDepth, -1.f, 1.f);
						curPixelDepth = static_cast<int>(tempDepth * INT_MAX);// avoid z-fighting	
						////Instead of using atmoicCAS, I use atomicMin which is much easier to implement
						int oldDepth = atomicMin(&depth[pixelIdx], curPixelDepth);
						if (curPixelDepth < oldDepth) {
							curFrag.eyePos = curBaryCord.x * triEyePos[0] + curBaryCord.y * triEyePos[1] + curBaryCord.z * triEyePos[2];
							curFrag.eyeNor = glm::normalize(curBaryCord.x * triEyeNor[0] + curBaryCord.y * triEyeNor[1] + curBaryCord.z * triEyeNor[2]);
							curFrag.dev_diffuseTex = targetPrimitive.v[0].dev_diffuseTex;
							curFrag.diffuseTexWidth = targetPrimitive.v[0].diffuseTexWidth;
							curFrag.diffuseTexHeight = targetPrimitive.v[0].diffuseTexHeight;
							//color without correction
							//fragment[pixelIdx].color = glm::vec3(1.f, 0.f, 0.f);
							//curFrag.color = curFrag.eyeNor;
							curFrag.color = curBaryCord.x * targetPrimitive.v[0].col + curBaryCord.y * targetPrimitive.v[1].col + curBaryCord.z * targetPrimitive.v[2].col;

#if PERSPECTIVE
							float z = curBaryCord.x*triEyePos[0].z + curBaryCord.y*triEyePos[1].z + curBaryCord.z*triEyePos[2].z;
							curFrag.texcoord0 = z*(curBaryCord.x / triEyePos[0].z*Texcoord[0] +
								curBaryCord.y / triEyePos[1].z*Texcoord[1] +
								curBaryCord.z / triEyePos[2].z*Texcoord[2]);
#else
							curFrag.texcoord0 = curBaryCord.x * Texcoord[0] + curBaryCord.y * Texcoord[1] + curBaryCord.z * Texcoord[2];
#endif

						}
					}
				}
			}
		}
		else if (renderMode == 2) {
			for (int i = 0; i < 3; i++) {
				int x = (int)tri[i].x;
				int y = (int)tri[i].y;
				if (x >= 0 && x <= width - 1 && y >= 0 && y <= height - 1) {
					pixelIdx = x + y * width;
					fragment[pixelIdx].color = glm::vec3(1.f, 0.8f, 0);
				}
			}
		}
		else if (renderMode == 1) {
			Bresenham(tri[0], tri[1], fragment, width, height);
			Bresenham(tri[1], tri[2], fragment, width, height);
			Bresenham(tri[0], tri[2], fragment, width, height);
		}
	}
}







/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal,int renderMode,float* timecount) {
    int sideLength2d = 8;
	float T = 0;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	dim3 numThreadsPerBlock(128);
	{
		curPrimitiveBeginId = 0;

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				/*T = 0.0f;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);*/

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				
				//cudaEventRecord(stop, 0);
				//cudaEventSynchronize(start);
				//cudaEventSynchronize(stop);
				//cudaEventElapsedTime(&T, start, stop);
				//timecount[0] += T;
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();

				/*T = 0.0f;
				cudaEventCreate(&start);
				cudaEventCreate(&stop);
				cudaEventRecord(start, 0);*/

				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);

				/*cudaEventRecord(stop, 0);
				cudaEventSynchronize(start);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&T, start, stop);
				timecount[1] += T;*/
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	dim3 numPrmitiveBlock((curPrimitiveBeginId + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	//backfaceCulling
	if (renderMode == 0) {
		backFaceCulling<< < numPrmitiveBlock, numThreadsPerBlock >> >(curPrimitiveBeginId, dev_primitives, dev_flag);
		CompressPrimitives(curPrimitiveBeginId, dev_primitives, dev_flag);
	}
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	// TODO: rasterize
	numPrmitiveBlock = dim3((curPrimitiveBeginId + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
	//T = 0.0f;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start, 0);

	kernRasterize << <numPrmitiveBlock, numThreadsPerBlock >> > (dev_fragmentBuffer, dev_depth, dev_primitives, curPrimitiveBeginId,width, height,renderMode);

	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(start);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&T, start, stop);
	//timecount[2] += T;

    // Copy depthbuffer colors into framebuffer
	/*T = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/

	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer,renderMode);

	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(start);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&T, start, stop);
	//timecount[3] += T;

	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
	/*T = 0.0f;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);*/
    
	sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, imageWidth, imageHeight, dev_framebuffer,SSAA);
	//
	//cudaEventRecord(stop, 0);
	//cudaEventSynchronize(start);
	//cudaEventSynchronize(stop);
	//cudaEventElapsedTime(&T, start, stop);
	//timecount[4] += T;

    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	cudaFree(dev_flag);
	dev_flag = NULL;
    checkCUDAError("rasterize Free");
}
