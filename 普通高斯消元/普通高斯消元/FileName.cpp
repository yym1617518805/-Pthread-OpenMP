/*
//这部分代码是使用Pthread进行普通高斯消元的优化。
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <math.h>
#include <pthread.h>
#include <immintrin.h>

#define MAX_THREADS 4 // 设置最大线程数
typedef long long LL;
using namespace std;

int N;
vector<vector<double>> f(2100, vector<double>(2100, 0));
double esp = 1e-6;

void input()
{
    cin >> N;
    for (int i = 1; i <= N; i++)
    {
        for (int j = i; j <= N + 1; j++)
        {
            if (j == i)
                f[i][j] = 1;
            else
                f[i][j] = 2;
        }
    }
    return;
}

int Count = 1;

struct ThreadData
{
    int start;
    int end;
};

void* solve_thread(void* arg)
{
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start; i <= data->end; i++)
    {
        __m256d factor = _mm256_set1_pd(f[i][Count]);
        for (int j = N + 1; j >= i+4; j -= 4) {
            __m256d row_values = _mm256_loadu_pd(&f[i][j - 3]);
            __m256d count_values = _mm256_loadu_pd(&f[Count][j - 3]);
            row_values = _mm256_fnmadd_pd(factor, count_values, row_values); // f[i][j] -= f[Count][j] * factor
            _mm256_storeu_pd(&f[i][j - 3], row_values);
        }
    }
    pthread_exit(NULL);
    return NULL;
}

int main()
{
    input();
    pthread_t threads[MAX_THREADS];
    ThreadData thread_data[MAX_THREADS];
    int thread_count = min(N, MAX_THREADS); // 实际线程数取决于矩阵行数和最大线程数的较小
    int rows_per_thread = (N - Count) / thread_count; // 每个线程处理的行数
    int remaining_rows = (N - Count) % thread_count;   // 余下的行数
    int start_row = Count + 1;

    for (int i = 1; i <= N - 1; i++) {
        if (fabs(f[i][i]) < esp) {
            for (int j = i + 1; j <= N; j++) {
                if (fabs(f[j][i]) > esp) {
                    swap(f[j], f[i]);
                    break;
                }
            }
        }

        __m256d factor_inv = _mm256_set1_pd(1.0 / f[i][i]);
        for (int j = N + 1; j >= i+4; j -= 4) {
            __m256d row_values = _mm256_loadu_pd(&f[i][j - 3]);
            row_values = _mm256_mul_pd(row_values, factor_inv); // f[i][j] /= f[i][i]
            _mm256_storeu_pd(&f[i][j - 3], row_values);
        }

        for (int i = 0; i < thread_count; i++) {
            thread_data[i].start = start_row;
            thread_data[i].end = start_row + rows_per_thread - 1 + (i < remaining_rows ? 1 : 0);
            pthread_create(&threads[i], NULL, solve_thread, (void*)&thread_data[i]);
            start_row = 1 + (++Count);
            rows_per_thread = (N - Count) / thread_count;
            remaining_rows = (N - Count) % thread_count;
        }

        for (int i = 0; i < thread_count; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    for (int k = N - 1; k >= 1; k--) {
        for (int j = N; j > k; j--) {
            f[k][N + 1] -= f[j][N + 1] * f[k][j];
        }
    }

    return 0;
}

*/

/*
//普通特殊高斯消元
#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>
#include<fstream>
#include <fstream>  // 用于文件输入输出
#include <string>   // 用于字符串处
#include<sstream>
#include<pthread.h>
using namespace std;
const int N = 400;
const int Len = 255;
vector<vector<int>> a(5, vector<int>(90000, 0));//消元子
int c[90000];

int String_to_int(string a) {
	int i = 0;
	int res = 0;
	for (i; i < a.length(); i++) {
		res *= 10;
		res += a[i] - '0';
	}
	return res;
}
string int_to_String(int a) {
	ostringstream os;
	os << a;
	return os.str();
}
vector<int> reca(5, 0);
void input(istringstream s, vector<int> q) {
	string st;
	while (s >> st) {
		q[String_to_int(st)] = 1;
	}
	return;
}
void inFile(string load, vector<int> s) {
	ofstream fil;
	fil.open(load, ios::app);
	bool flag = false;
	for (int i = Len; i >= 0; i--) {
		if (s[i]) {
			if (!flag)   c[i] = 1;
			flag = true;
			fil << int_to_String(i) << " ";
		}
	}

	if (!flag) {
		if (load == "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt")
			fil << endl;
		return;
	}
	fil << endl;
	fil.close();
	return;
}

vector<int> xiaoyuan(vector<int>s, vector<int>q) {
	for (int i = Len; i >= 0; i--) {
		s[i] = s[i] ^ q[i];
	}
	return s;
}
int  signal[2];
void get_duijiaoxian(int s[]) {

}
void get_xyz() {

}
int main() {

	//文件的初始化
	std::ofstream ofs1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs1.close(); // 关闭文件流
	std::ofstream ofs2("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs2.close(); // 关闭文件流
	std::ofstream ofs3("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs3.close(); // 关闭文件流

	std::string sourceFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//被消元行.txt";
	// 目标文件路径
	std::string targetFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt";
	// 打开源文件进行读取
	std::ifstream source(sourceFile, std::ios::binary);
	// 打开目标文件进行写入（如果不存在会创建，存在则覆盖）
	std::ofstream target(targetFile, std::ios::binary);
	// 检查文件是否成功打开
	if (!source.is_open() || !target.is_open()) {
		std::cerr << "Error: Could not open files." << std::endl;
		return 1;
	}
	// 使用流的拷贝操作，将源文件内容复制到目标文件
	target << source.rdbuf();
	// 关闭文件流
	source.close();
	target.close();



	ifstream file("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//消元子 - 副本.txt");
	string line;
	int i = 0;
	int curfile = 1;
	string curFile = "res" + int_to_String(curfile) + ".txt";
	ofstream fileoutres;//最终的结果。
	fileoutres.open("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", ios::app);
	ifstream fileout("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt");  //被消元
	ifstream fileout1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt");
	bool flagg = true;
	while (flagg) {
		ifstream fileout("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt");  //被消元
		ifstream fileout1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt");
		int num = 0;
		int num1 = 0;
		flagg = true;
		int needle = 0;
		while (a.size() > 5) {
			a.pop_back();
		}
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j <= Len; j++) {
				a[i][j] = 0;
			}
		}
		while (needle < 5 && getline(file, line)) {
			string str;
			istringstream stream(line);
			int flag = false;
			while (stream >> str) {
				if (!flag) {
					reca[needle] = String_to_int(str);
					flag = true;
				}
				a[needle][String_to_int(str)] = 1;
			}
			needle++;
		}
		int p = 0;
		while (p < signal[curfile]) {
			getline(curfile == 1 ? fileout : fileout1, line);
			p++;
		}
		while (getline(curfile == 1 ? fileout : fileout1, line)) {// 从文件中读取一行
			signal[curfile]++;
			flagg = false;
			int start = 0;
			string str;
			istringstream stream(line);
			vector<int> b(90000, 0);
			bool flag = true;
			while (stream >> str) {
				if (flag) {
					start = String_to_int(str);
					flag = false;
				}
				b[String_to_int(str)] = 1;  //读取被消元素
			}
			flag = false;
			for (int i = 0; i < a.size(); i++) {
				if (start > reca[i]) {
					flag = true;
					a.insert(a.begin() + i, b);
					reca.insert(reca.begin() + i, start);
					inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", b);
					break;
				}
				else if (start < reca[i]) {
					continue;
				}
				else if (start == reca[i]) {
					b = xiaoyuan(b, a[i]);
					for (start; start >= 0; start--) {
						if (b[start]) {
							break;
						}
					}
					continue;
				}
			}
			if (!flag) {
				num1++;
				string curF = "res" + int_to_String(curfile ^ 1) + ".txt";
				inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//" + curF, b);
			}
		}
		curfile = curfile ^ 1;
		if (flagg) {
			break;
		}
		fileout.close();
		fileout1.close();
		flagg = true;
	}
	fileout.close();
	fileout1.close();
	fileoutres.close();
	return 0;
}

/*
/*
//SSE+Pthread
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <emmintrin.h> // 包含SSE2头文件
#include <time.h>      // 用于计时

#define NUM_THREADS 4
#define N 2000

double A[N][N];
double b[N];
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct {
	int t_id;
} threadParam_t;

void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = rand() % 100; // 为了避免除以0, 生成随机数在一个范围内
		}
		b[i] = rand() % 100;
	}
}

void *threadFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k) {
		if (t_id == 0) {
			double div_val = A[k][k];
			__m128d div_vec = _mm_set1_pd(div_val);
			for (int j = k + 1; j < N; j++) {
				A[k][j] /= div_val;
			}
			b[k] /= div_val;
			A[k][k] = 1.0;
		}

		pthread_barrier_wait(&barrier_Division);

		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
			double factor = A[i][k];
			__m128d factor_vec = _mm_set1_pd(factor);

			for (int j = k + 1; j < N; j++) {
				A[i][j] -= factor * A[k][j];
			}
			b[i] -= factor * b[k];
			A[i][k] = 0.0;
		}

		pthread_barrier_wait(&barrier_Elimination);
	}

	pthread_exit(NULL); // 正确地返回NULL
	return NULL;
}



int main() {
	// 初始化barrier
	pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

	m_reset();

	// 记录开始时间
	clock_t start = clock();

	// 创建线程
	pthread_t handles[NUM_THREADS];
	threadParam_t param[NUM_THREADS];

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handles[t_id], NULL);
	}

	// 记录结束时间
	clock_t end = clock();

	// 计算运行时间
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken: %f seconds\n", time_spent);

	// 销毁所有的barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);

	return 0;
}

*/

/*

//普通高斯消元
#include <stdio.h>
#include <stdlib.h>

#define N 2000 // 矩阵大小

double A[N][N]; // 系数矩阵
double b[N];    // 常数向量
double y[N];    // 解向量

void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[j][i] = rand(); // 生成随机的系数矩阵 A
		}
		b[i] = rand(); // 生成随机的常数向量 b
	}
}

int main() {
	// 初始化矩阵 A 和向量 b
	m_reset();

	// 消去过程
	for (int k = 0; k < N - 1; ++k) {
		for (int i = k + 1; i < N; ++i) {
			double factor = A[i][k] / A[k][k];
			for (int j = k + 1; j < N; ++j) {
				A[i][j] -= factor * A[k][j];
			}
			b[i] -= factor * b[k];
		}
	}

	// 回代过程
	for (int i = 0; i < N; ++i) {
		y[i] = 0.0;
	}
	for (int i = N - 1; i >= 0; --i) {
		double sum = b[i];
		for (int j = i + 1; j < N; ++j) {
			sum -= A[i][j] * y[j];
		}
		y[i] = sum / A[i][i];
	}


	return 0;
}*/

/*
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>


#define MAX_THREADS 4 // 设置最大线程数
#define N 50 // 示例矩阵大小，可根据需要修改

typedef struct {
	int k; // 消去的轮次
	int t_id; // 线程 id
} threadParam_t;

double A[N][N];
double b[N];
pthread_barrier_t barrier;

void initialize_matrix() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = rand() % 100 + 1;
		}
		b[i] = rand() % 100 + 1;
	}
}

void* threadFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int k = p->k; // 消去的轮次
	int t_id = p->t_id; // 线程编号

	for (int i = k + t_id + 1; i < N; i += MAX_THREADS) { // 动态分配任务
		for (int j = k + 1; j < N; ++j) {
			A[i][j] = A[i][j] - A[i][k] * A[k][j];
		}
		b[i] = b[i] - A[i][k] * b[k];
		A[i][k] = 0;
	}

	pthread_exit(NULL);
	return NULL;
}

int main() {
	pthread_barrier_init(&barrier, NULL, MAX_THREADS);
	initialize_matrix();
	for (int k = 0; k < N; ++k) {
		// 主线程做除法操作
		for (int j = k + 1; j < N; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		b[k] = b[k] / A[k][k];
		A[k][k] = 1.0;

		// 创建工作线程，进行消去操作
		int worker_count = N - 1 - k; // 工作线程数量
		if (worker_count > MAX_THREADS) {
			worker_count = MAX_THREADS;
		}

		// 动态分配内存
		pthread_t* handles = (pthread_t*)malloc(worker_count * sizeof(pthread_t));
		threadParam_t* param = (threadParam_t*)malloc(worker_count * sizeof(threadParam_t));

		// 分配任务
		for (int t_id = 0; t_id < worker_count; t_id++) {
			param[t_id].k = k;
			param[t_id].t_id = t_id;
		}

		// 创建线程
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
		}

		// 主线程挂起等待所有的工作线程完成此轮消去工作
		for (int t_id = 0; t_id < worker_count; t_id++) {
			pthread_join(handles[t_id], NULL);
		}

		// 释放动态分配的内存
		free(handles);
		free(param);
	}

	pthread_barrier_destroy(&barrier);

	return 0;
}

*/

/*
按行访问Pthread并行优化
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <emmintrin.h> // Include SSE2 header
#include <time.h>      // For timing

#define NUM_THREADS 4
#define N 2000

double A[N][N];
double b[N];
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct {
	int t_id;
} threadParam_t;

void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[j][i] = rand() % 100; // 为了避免除以 0, 生成随机数在一个范围内
		}
		b[i] = rand() % 100;
	}
}

void* threadFunc(void* param) {
	threadParam_t* p = (threadParam_t*)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k) {
		if (t_id == 0) {
			double div_val = A[k][k];
			__m128d div_vec = _mm_set1_pd(div_val);
			for (int j = k + 1; j <= N - 2; j += 2) {
				__m128d row_vec = _mm_loadu_pd(&A[k][j]);
				row_vec = _mm_div_pd(row_vec, div_vec);
				_mm_storeu_pd(&A[k][j], row_vec);
			}
			for (int j = (N / 2) * 2; j < N; j++) {
				A[k][j] /= div_val;
			}
			b[k] /= div_val;
			A[k][k] = 1.0;
		}

		pthread_barrier_wait(&barrier_Division);

		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
			double factor = A[i][k];
			__m128d factor_vec = _mm_set1_pd(factor);

			for (int j = k + 1; j <= N - 2; j += 2) {
				__m128d row_k = _mm_loadu_pd(&A[k][j]);
				__m128d row_i = _mm_loadu_pd(&A[i][j]);
				row_i = _mm_sub_pd(row_i, _mm_mul_pd(factor_vec, row_k));
				_mm_storeu_pd(&A[i][j], row_i);
			}

			for (int j = (N / 2) * 2; j < N; j++) {
				A[i][j] -= factor * A[k][j];
			}

			b[i] -= factor * b[k];
			A[i][k] = 0.0;
		}

		pthread_barrier_wait(&barrier_Elimination);
	}

	pthread_exit(NULL);
	return NULL;
}

int main() {
	// 初始化 barrier
	pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

	m_reset();

	// 记录开始时间
	clock_t start = clock();

	// 创建线程
	pthread_t handles[NUM_THREADS];
	threadParam_t param[NUM_THREADS];

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void*)&param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handles[t_id], NULL);
	}

	// 记录结束时间
	clock_t end = clock();

	// 计算运行时间
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken: %f seconds\n", time_spent);

	// 销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);

	return 0;
}*/



/*
//列访问SIMD
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <emmintrin.h> // Include SSE2 header
#include <time.h>      // For timing

#define NUM_THREADS 4
#define N 2000

double A[N][N];
double b[N];
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;

typedef struct {
	int t_id;
} threadParam_t;

void m_reset() {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			A[j][i] = rand() % 100; // 为了避免除以 0, 生成随机数在一个范围内
		}
		b[i] = rand() % 100;
	}
}

void *threadFunc(void *param) {
	threadParam_t *p = (threadParam_t *)param;
	int t_id = p->t_id;

	for (int k = 0; k < N; ++k) {
		if (t_id == 0) {
			double div_val = A[k][k];
			__m128d div_vec = _mm_set1_pd(div_val);
			for (int j = k + 1; j <= N - 2; j += 2) {
				__m128d row_vec = _mm_loadu_pd(&A[k][j]);
				row_vec = _mm_div_pd(row_vec, div_vec);
				_mm_storeu_pd(&A[k][j], row_vec);
			}
			for (int j = (N / 2) * 2; j < N; j++) {
				A[k][j] /= div_val;
			}
			b[k] /= div_val;
			A[k][k] = 1.0;
		}

		pthread_barrier_wait(&barrier_Division);

		for (int i = k + 1 + t_id; i < N; i += NUM_THREADS) {
			double factor = A[i][k];
			__m128d factor_vec = _mm_set1_pd(factor);

			for (int j = k + 1; j <= N - 2; j += 2) {
				__m128d row_k = _mm_loadu_pd(&A[k][j]);
				__m128d row_i = _mm_loadu_pd(&A[i][j]);
				row_i = _mm_sub_pd(row_i, _mm_mul_pd(factor_vec, row_k));
				_mm_storeu_pd(&A[i][j], row_i);
			}

			for (int j = (N / 2) * 2; j < N; j++) {
				A[i][j] -= factor * A[k][j];
			}

			b[i] -= factor * b[k];
			A[i][k] = 0.0;
		}

		pthread_barrier_wait(&barrier_Elimination);
	}

	pthread_exit(NULL);
	return NULL;
}

int main() {
	// 初始化 barrier
	pthread_barrier_init(&barrier_Division, NULL, NUM_THREADS);
	pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);

	m_reset();

	// 记录开始时间
	clock_t start = clock();

	// 创建线程
	pthread_t handles[NUM_THREADS];
	threadParam_t param[NUM_THREADS];

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		param[t_id].t_id = t_id;
		pthread_create(&handles[t_id], NULL, threadFunc, (void *)&param[t_id]);
	}

	for (int t_id = 0; t_id < NUM_THREADS; t_id++) {
		pthread_join(handles[t_id], NULL);
	}

	// 记录结束时间
	clock_t end = clock();

	// 计算运行时间
	double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
	printf("Time taken: %f seconds\n", time_spent);

	// 销毁所有的 barrier
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);

	return 0;
}
*/

/*
* 
* //使用openMP进行并行化消元
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h> // SSE intrinsics

#define NUM_THREADS 4

void gaussian_elimination(double** mat, double* b, int n) {
	int i, j, k;
	double tmp;

#pragma omp parallel if(n > 1) num_threads(NUM_THREADS) private(i, j, k, tmp)
	{
		for (k = 0; k < n; ++k) {
#pragma omp single
			{
				tmp = mat[k][k];
				if (tmp == 0.0) {
					printf("Error: Singular matrix detected at iteration %d\n", k);
					return;
				}
				for (j = k + 1; j < n; ++j) {
					mat[k][j] /= tmp;
				}
				mat[k][k] = 1.0;
				b[k] /= tmp;
			}

#pragma omp for schedule(static)
			for (i = k + 1; i < n; ++i) {
				tmp = mat[i][k];
				__m128d tmp_vec = _mm_set1_pd(tmp); // Load tmp into a SSE vector
				for (j = k + 1; j < n; j += 2) { // Process two elements at a time
					__m128d mat_ij = _mm_loadu_pd(&mat[i][j]); // Load two elements of mat[i][j] into a SSE vector
					__m128d mat_kj = _mm_loadu_pd(&mat[k][j]); // Load two elements of mat[k][j] into a SSE vector
					__m128d tmp_result = _mm_mul_pd(mat_kj, tmp_vec); // Multiply mat[k][j] by tmp
					tmp_result = _mm_sub_pd(mat_ij, tmp_result); // Subtract tmp_result from mat[i][j]
					_mm_storeu_pd(&mat[i][j], tmp_result); // Store the result back to mat[i][j]
				}
				mat[i][k] = 0.0;
				b[i] -= tmp * b[k];
			}
		}
	}
}

int main() {
	int n = 500; // Example matrix size
	double** mat;
	double* b;

	// Allocate memory for the matrix
	mat = (double**)malloc(n * sizeof(double*));
	for (int i = 0; i < n; i++) {
		mat[i] = (double*)malloc(n * sizeof(double));
	}

	// Allocate memory for the vector b
	b = (double*)malloc(n * sizeof(double));

	// Initialize the matrix and vector b (example values)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			mat[i][j] = rand() % 100 + 1; // Example initialization
		}
		b[i] = rand() % 100 + 1; // Example initialization
	}

	// Measure the execution time
	double start_time = omp_get_wtime();
	gaussian_elimination(mat, b, n);
	double end_time = omp_get_wtime();

	printf("Time taken for Gaussian Elimination: %f seconds\n", end_time - start_time);

	// Free the allocated memory
	for (int i = 0; i < n; i++) {
		free(mat[i]);
	}
	free(mat);
	free(b);

	return 0;
}
*/

/*
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <pthread.h>
#include <immintrin.h> // 引入AVX指令集

typedef long long LL;
using namespace std;

const int MAX_SIZE = 2010;
double esp = 1e-6;
const int THREAD_COUNT = 4;
int N = 50;
vector<vector<double>> f(2020, vector<double>(2020+ 1, 0));
struct ThreadData {
	vector<vector<double>>* f;
	int i;
	int n;
	int k_start;
	int k_end;
};
void input()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = i; j <= N + 1; j++)
		{
			if (j == i)
				f[i][j] = 1;
			else
				f[i][j] = 2;
		}
	}
	return;
}


// 规范化一行的任务
void* normalize_row(void* arg) {
	ThreadData* data = (ThreadData*)arg;
	vector<vector<double>>& f = *(data->f);
	int i = data->i;
	int n = data->n;

	double pivot = f[i][i];
	__m256d pivot_vec = _mm256_set1_pd(pivot);


	for (int j = i; j <= n + 1; j += 4) {
		__m256d row_vec = _mm256_loadu_pd(&f[i][j]);
		__m256d result_vec = _mm256_div_pd(row_vec, pivot_vec);
		_mm256_storeu_pd(&f[i][j], result_vec);
	}
	return nullptr;
}

// 消去下面行的任务
void* eliminate_rows(void* arg) {
	ThreadData* data = (ThreadData*)arg;
	vector<vector<double>>& f = *(data->f);
	int i = data->i;
	int k_start = data->k_start;
	int k_end = data->k_end;
	int n = data->n;

	for (int k = k_start; k <= k_end; k++) {
		double scale = f[k][i];
		__m256d scale_vec = _mm256_set1_pd(scale);

		for (int j = i; j <= n + 1; j += 4) {
			__m256d row_i_vec = _mm256_loadu_pd(&f[i][j]);
			__m256d row_k_vec = _mm256_loadu_pd(&f[k][j]);
			__m256d subtract_vec = _mm256_mul_pd(row_i_vec, scale_vec);
			__m256d result_vec = _mm256_sub_pd(row_k_vec, subtract_vec);
			_mm256_storeu_pd(&f[k][j], result_vec);
		}
	}
	return nullptr;
}

// 使用AVX优化和Pthread进行高斯消元
int solve(vector<vector<double>>& f, int n) {
	for (int i = 1; i <= n; i++) {
		int r = i;
		for (int k = i; k <= n; k++) {
			if (fabs(f[k][i]) > esp) { r = k; break; }
		}

		if (r != i) {
			swap(f[i], f[r]);
		}

		if (fabs(f[i][i]) < esp) {
			return 0; // 无解
		}

		pthread_t normalize_thread;
		ThreadData normalize_data = { &f, i, n, 0, 0 };
		pthread_create(&normalize_thread, nullptr, normalize_row, &normalize_data);
		pthread_join(normalize_thread, nullptr);
		
		pthread_t threads[THREAD_COUNT];
		ThreadData thread_data[THREAD_COUNT];
		int rows_per_thread = (n - i) / THREAD_COUNT;

		for (int t = 0; t < THREAD_COUNT; t++) {
			int k_start = i + 1 + t * rows_per_thread;
			int k_end = (t == THREAD_COUNT - 1) ? n : k_start + rows_per_thread - 1;
			thread_data[t] = { &f, i, n, k_start, k_end };
			pthread_create(&threads[t], nullptr, eliminate_rows, &thread_data[t]);
		}
		for (int t = 0; t < THREAD_COUNT; t++) {
			pthread_join(threads[t], nullptr);
		}
	}

	// 回代求解
	for (int k = n - 1; k >= 1; k--) {
		for (int j = n; j > k; j--) {
			f[k][n + 1] -= f[j][n + 1] * f[k][j];
		}
	}

	return 1; // 找到解
}

int main() {
	input();



	if (solve(f, N)) {
	}
	else {
		cout << "无解" << endl;
	}

	return 0;
}


*/
/*
//Pthread并行消元(特殊高斯消元)
#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>
#include<fstream>
#include <fstream>  // 用于文件输入输出
#include <string>   // 用于字符串处
#include<sstream>
#include<pthread.h>
using namespace std;
const int N = 400;
const int Len = 255;
vector<vector<int>> a(5, vector<int>(90000, 0));//消元子
int c[90000];
pthread_mutex_t lock;
struct ThreadData {
	int* start;
	vector<int> b;
	bool* flag;
	int* num1;
	int curfile;
};
int String_to_int(string a) {
	int i = 0;
	int res = 0;
	for (i; i < a.length(); i++) {
		res *= 10;
		res += a[i] - '0';
	}
	return res;
}
string int_to_String(int a) {
	ostringstream os;
	os << a;
	return os.str();
}
vector<int> reca(5, 0);
void input(istringstream s, vector<int> q) {
	string st;
	while (s >> st) {
		q[String_to_int(st)] = 1;
	}
	return;
}

void inFile(string load, vector<int> s) {
	ofstream fil;
	fil.open(load, ios::app);
	bool flag = false;
	for (int i = Len; i >= 0; i--) {
		if (s[i]) {
			if (!flag)   c[i] = 1;
			flag = true;
			fil << int_to_String(i) << " ";
		}
	}

	if (!flag) {
		if (load == "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt")
			fil << endl;
		return;
	}
	fil << endl;
	fil.close();
	return;
}

vector<int> xiaoyuan(vector<int>s, vector<int>q) {
	for (int i = Len; i >= 0; i--) {
		s[i] = s[i] ^ q[i];
	}
	return s;
}
int  signal[2];
void* process_line(void* arg) {
	ThreadData* data = (ThreadData*)arg;
	int* start1 = data->start;
	int start = *start1;
	vector<int> b = data->b;
	bool* flag1 = data->flag;
	bool flag = *flag1;
	int* num1 = data->num1;
	int curfile = data->curfile;
	for (int i = 0; i < a.size(); i++) {
		if (start > reca[i]) {
			pthread_mutex_lock(&lock);
			flag = true;
			a.insert(a.begin() + i, b);
			reca.insert(reca.begin() + i, start);
			inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", b);
			break;
		}
		else if (start < reca[i]) {
			continue;
		}
		else if (start == reca[i]) {
			continue;
			b = xiaoyuan(b, a[i]);
			for (start; start >= 0; start--) {
				if (b[start]) break;
			}
		}
	}
	if (!flag) {
		num1++;
		string curF = "res" + int_to_String(curfile ^ 1) + ".txt";
		inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//" + curF, b);
	}

	*flag1 = flag;
	*start1 = start;
	return NULL;
	pthread_mutex_unlock(&lock);
}
int main() {

	//文件的初始化
	std::ofstream ofs1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs1.close(); // 关闭文件流
	std::ofstream ofs2("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs2.close(); // 关闭文件流
	std::ofstream ofs3("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", std::ios::trunc); // 使用 std::ios::trunc 标志来清空文件
	ofs3.close(); // 关闭文件流

	std::string sourceFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//被消元行.txt";
	// 目标文件路径
	std::string targetFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt";
	// 打开源文件进行读取
	std::ifstream source(sourceFile, std::ios::binary);
	// 打开目标文件进行写入（如果不存在会创建，存在则覆盖）
	std::ofstream target(targetFile, std::ios::binary);
	// 检查文件是否成功打开
	if (!source.is_open() || !target.is_open()) {
		std::cerr << "Error: Could not open files." << std::endl;
		return 1;
	}
	// 使用流的拷贝操作，将源文件内容复制到目标文件
	target << source.rdbuf();
	// 关闭文件流
	source.close();
	target.close();



	ifstream file("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//消元子 - 副本.txt");
	string line;
	int i = 0;
	int curfile = 1;
	string curFile = "res" + int_to_String(curfile) + ".txt";
	ofstream fileoutres;//最终的结果。
	fileoutres.open("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", ios::app);
	ifstream fileout("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt");  //被消元
	ifstream fileout1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt");
	bool flagg = true;
	while (flagg) {
		ifstream fileout("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt");  //被消元
		ifstream fileout1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt");
		int num = 0;
		int num1 = 0;
		flagg = true;
		int needle = 0;
		while (a.size() > 5) {
			a.pop_back();
		}
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j <= Len; j++) {
				a[i][j] = 0;
			}
		}
		while (needle < 5 && getline(file, line)) {
			string str;
			istringstream stream(line);
			int flag = false;
			while (stream >> str) {
				if (!flag) {
					reca[needle] = String_to_int(str);
					flag = true;
				}
				a[needle][String_to_int(str)] = 1;
			}
			needle++;
		}
		int p = 0;
		while (p < signal[curfile]) {
			getline(curfile == 1 ? fileout : fileout1, line);
			p++;
		}
		while (getline(curfile == 1 ? fileout : fileout1, line)) {
			signal[curfile]++;
			flagg = false;
			int start = 0;
			string str;
			istringstream stream(line);
			vector<int> b(90000, 0);
			bool flag = true;
			while (stream >> str) {
				if (flag) {
					start = String_to_int(str);
					flag = false;
				}
				b[String_to_int(str)] = 1;
			}
			flag = false;

			pthread_t thread;
			ThreadData data = { &start, b, &flag ,&num1,curfile};
			pthread_create(&thread, nullptr, process_line, &data);
			pthread_join(thread, nullptr);
		}
		curfile = curfile ^ 1;
		if (flagg) {
			break;
		}
		fileout.close();
		fileout1.close();
		flagg = true;
	}
	fileout.close();
	fileout1.close();
	fileoutres.close();
	return 0;
}


*/


#include <immintrin.h> // AVX指令集
#include <omp.h> // OpenMP
#include<iostream>
#include<iomanip>
#include<vector>
#include<cmath>
#include<fstream>
#include<string>
#include<sstream>

using namespace std;

const int N = 2000;
const int Len = 255;
vector<vector<int>> a(5, vector<int>(90000, 0)); // 消元子
int c[90000];

int String_to_int(const string& a) {
	int res = 0;
	for (char ch : a) {
		res = res * 10 + (ch - '0');
	}
	return res;
}

string int_to_String(int a) {
	ostringstream os;
	os << a;
	return os.str();
}

vector<int> reca(5, 0);

void input(istringstream& s, vector<int>& q) {
	string st;
	while (s >> st) {
		q[String_to_int(st)] = 1;
	}
}

void inFile(const string& load, const vector<int>& s) {
	ofstream fil;
	fil.open(load, ios::app);
	bool flag = false;
	for (int i = Len; i >= 0; i--) {
		if (s[i]) {
			if (!flag) c[i] = 1;
			flag = true;
			fil << int_to_String(i) << " ";
		}
	}

	if (!flag) {
		if (load == "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt")
			fil << endl;
		return;
	}
	fil << endl;
	fil.close();
}

vector<int> xiaoyuan(const vector<int>& s, const vector<int>& q) {
	vector<int> result(s.size(), 0);
#pragma omp parallel for
	for (size_t i = 0; i < s.size(); i += 8) { // 每次处理8个32位整数
		__m256i s_vec = _mm256_loadu_si256((__m256i*) & s[i]); // 加载AVX向量
		__m256i q_vec = _mm256_loadu_si256((__m256i*) & q[i]);
		__m256i result_vec = _mm256_xor_si256(s_vec, q_vec); // 执行按位异或
		_mm256_storeu_si256((__m256i*) & result[i], result_vec); // 存储结果
	}
	return result;
}

int signal[2];

void get_duijiaoxian(int s[]) {
	// Implementation not provided
}

void get_xyz() {
	// Implementation not provided
}

int main() {
	// 文件的初始化
	std::ofstream ofs1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt", std::ios::trunc);
	ofs1.close();
	std::ofstream ofs2("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt", std::ios::trunc);
	ofs2.close();
	std::ofstream ofs3("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", std::ios::trunc);
	ofs3.close();

	std::string sourceFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//被消元行.txt";
	std::string targetFile = "D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt";
	std::ifstream source(sourceFile, std::ios::binary);
	std::ofstream target(targetFile, std::ios::binary);

	if (!source.is_open() || !target.is_open()) {
		std::cerr << "Error: Could not open files." << std::endl;
		return 1;
	}

	target << source.rdbuf();
	source.close();
	target.close();

	ifstream file("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//消元子 - 副本.txt");
	string line;
	int curfile = 1;
	ofstream fileoutres("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", ios::app);

	bool flagg = true;
	while (flagg) {
		ifstream fileout("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res1.txt");
		ifstream fileout1("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res0.txt");
		int num1 = 0;
		flagg = true;
		int needle = 0;
		while (a.size() > 5) {
			a.pop_back();
		}
		for (int i = 0; i < 5; i++) {
			fill(a[i].begin(), a[i].end(), 0);
		}
		while (needle < 5 && getline(file, line)) {
			string str;
			istringstream stream(line);
			bool flag = false;
			while (stream >> str) {
				if (!flag) {
					reca[needle] = String_to_int(str);
					flag = true;
				}
				a[needle][String_to_int(str)] = 1;
			}
			needle++;
		}

		int p = 0;
		while (p < signal[curfile]) {
			getline(curfile == 1 ? fileout : fileout1, line);
			p++;
		}

#pragma omp parallel
		{
			vector<int> b(90000, 0);
#pragma omp for schedule(dynamic)
			for (int i = 0; i < 5; i++) {
				if (getline(curfile == 1 ? fileout : fileout1, line)) {
					signal[curfile]++;
					flagg = false;
					int start = 0;
					string str;
					istringstream stream(line);
					bool flag = true;
					while (stream >> str) {
						if (flag) {
							start = String_to_int(str);
							flag = false;
						}
						b[String_to_int(str)] = 1;  // 读取被消元素
					}
					flag = false;
					for (int i = 0; i < a.size(); i++) {
						if (start > reca[i]) {
							flag = true;
							a.insert(a.begin() + i, b);
							reca.insert(reca.begin() + i, start);
							inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//res2.txt", b);
							break;
						}
						else if (start < reca[i]) {
							continue;
						}
						else if (start == reca[i]) {
							b = xiaoyuan(b, a[i]);
							for (; start >= 0; start--) {
								if (b[start]) {
									break;
								}
							}
							continue;
						}
					}
					if (!flag) {
						num1++;
						string curF = "res" + int_to_String(curfile ^ 1) + ".txt";
						inFile("D://Gusee//Groebner//测试样例2 矩阵列数254，非零消元子106，被消元行53//" + curF, b);
					}
				}
			}
		}

		curfile = curfile ^ 1;
		if (flagg) {
			break;
		}
		fileout.close();
		fileout1.close();
		flagg = true;
	}

	return 0;
}
