static int a[10];
int n = 10;

int main() {  
  int s = 0;

  for (int i = 0; i < n; ++i) {
    a[i] = i;
  }
  a[0]=100;
  for (int j = 0; j < n; ++j) {
    s += a[j];
  }
  return s;
}