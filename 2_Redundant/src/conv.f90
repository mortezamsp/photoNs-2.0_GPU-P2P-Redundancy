
! Interface to convolution by 2decomp
! Q. Wang 2017 Mar 9
subroutine get_local_size(global_size, vproc, local_start, local_end, local_size)
  use types
  use decomp_2d_constants
  use decomp_2d_mpi
  use decomp_2d
  implicit none

  integer, dimension(2), intent(in) :: vproc
  integer, dimension(3), intent(in) :: global_size
  integer, dimension(3), intent(out) :: local_start, local_end, local_size

  integer :: ii, jj, kk, pc, pr
  ii = global_size(1)
  jj = global_size(2)
  kk = global_size(3)
  pc = vproc(1)
  pr = vproc(2)

  ! Initialize 2decomp
  call decomp_2d_init(ii, jj, kk, pc, pr)

  
  ! Get local domain information
  local_start = xstart - 1
  local_end = xend - 1
  local_size = xsize

  return
end subroutine get_local_size



subroutine conv_pmonly(data, nside, param)
  use decomp_2d_constants
  use decomp_2d_mpi
  use decomp_2d
  use decomp_2d_fft
  implicit none

  integer :: i, j, k, l, m, n, nhalf
  integer, dimension(3), intent(in) :: nside
  integer, dimension(3) :: fft_start, fft_end, fft_size
  real*8, dimension(xsize(1)*xsize(2)*xsize(3)), intent(inout) :: data
  real*8, dimension(2), intent(in) :: param
  real*8, parameter :: M_PI = 3.141592653589793238462643383
  real(mytype) :: k2, smooth, box, ismth2, gf, pref
  real(mytype), allocatable, dimension(:,:,:) :: in, in2
  complex(mytype), allocatable, dimension(:,:,:) :: out

  ! Initialize 2decomp FFT
  call decomp_2d_fft_init()

  ! Allocate input array
  allocate(in(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)))

  ! Copy data to input array
  do k = 0, xsize(3) - 1
    do j = 0, xsize(2) - 1
      do i = 0, xsize(1) - 1
        in(i + xstart(1), j + xstart(2), k + xstart(3)) = data((i * xsize(2) + j) * xsize(3) + k + 1)
      end do
    end do
  end do

  ! Get FFT size
  call decomp_2d_fft_get_size(fft_start, fft_end, fft_size)

  ! Allocate output array
  allocate(out(fft_start(1):fft_end(1), fft_start(2):fft_end(2), fft_start(3):fft_end(3)))

  ! Perform 3D FFT
  call decomp_2d_fft_3d(in, out)

  ! Green function
  nhalf = nside(1) / 2
  smooth = param(1)
  box = param(2)
  ismth2 = 2 * M_PI * smooth / box
  ismth2 = ismth2 * ismth2
  pref = box * box / (M_PI * nside(1) * nside(2) * nside(3))

  ! Apply Green function in Fourier space
  do k = fft_start(3), fft_end(3)
    n = k - 1
    if (n > nhalf) then
      n = n - nside(3)
    end if

    do j = fft_start(2), fft_end(2)
      m = j - 1
      if (m > nhalf) then
        m = m - nside(2)
      end if

      do i = fft_start(1), fft_end(1)
        l = i - 1
        if (l > nhalf) then
          l = l - nside(1)
        end if

        k2 = real(n * n + m * m + l * l, kind=mytype)

        gf = pref / k2

        if (l == 0 .and. m == 0 .and. n == 0) then
          gf = pref
        end if

        out(i, j, k) = out(i, j, k) * gf
      end do
    end do
  end do

  ! Allocate output array for inverse FFT
  allocate(in2(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)))

  ! Perform inverse 3D FFT
  call decomp_2d_fft_3d(out, in2)

  ! Copy result back to data
  do k = 0, xsize(3) - 1
    do j = 0, xsize(2) - 1
      do i = 0, xsize(1) - 1
        data((i * xsize(2) + j) * xsize(3) + k + 1) = in2(i + xstart(1), j + xstart(2), k + xstart(3))
      end do
    end do
  end do

  ! Deallocate arrays
  deallocate(in, in2, out)

  ! Finalize 2decomp FFT
  call decomp_2d_fft_finalize()

  return
end subroutine conv_pmonly

subroutine convolution(data, nside, param)
  use decomp_2d_constants
  use decomp_2d_mpi
  use decomp_2d
  use decomp_2d_fft
  implicit none

  integer :: i, j, k, l, m, n, nhalf
  integer, dimension(3), intent(in) :: nside
  integer, dimension(3) :: fft_start, fft_end, fft_size
  real*8, dimension(xsize(1)*xsize(2)*xsize(3)), intent(inout) :: data
  real*8, dimension(2), intent(in) :: param
  real*8, parameter :: M_PI = 3.1415926
  real(mytype) :: k2, smooth, box, ismth2, gf, pref, fx, fy, fz, ff
  real(mytype), allocatable, dimension(:,:,:) :: in, in2
  complex(mytype), allocatable, dimension(:,:,:) :: out

  call decomp_2d_fft_init

  allocate(in(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)))

  do k = 0, xsize(3) - 1
    do j = 0, xsize(2) - 1
      do i = 0, xsize(1) - 1
        in(i + xstart(1), j + xstart(2), k + xstart(3)) = data(((i) * xsize(2) + j) * xsize(3) + k + 1)
      end do
    end do
  end do

  call decomp_2d_fft_get_size(fft_start, fft_end, fft_size)

  allocate(out(fft_start(1):fft_end(1), fft_start(2):fft_end(2), fft_start(3):fft_end(3)))

  call decomp_2d_fft_3d(in, out)

  nhalf = nside(1) / 2
  smooth = param(1)
  box = param(2)
  ismth2 = 2 * M_PI * smooth / box
  ismth2 = ismth2 * ismth2
  pref = box * box / (M_PI * nside(1) * nside(2) * nside(3))

  do k = fft_start(3), fft_end(3)
    n = k - 1
    if (n > nhalf) then
      n = n - nside(3)
    end if

    fz = M_PI * n / nside(3)
    fz = sin(fz) / fz

    if (n == 0) then
      fz = 1
    end if

    do j = fft_start(2), fft_end(2)
      m = j - 1
      if (m > nhalf) then
        m = m - nside(2)
      end if

      fy = M_PI * m / nside(2)
      fy = sin(fy) / fy

      if (m == 0) then
        fy = 1
      end if

      do i = fft_start(1), fft_end(1)
        l = i - 1
        if (l > nhalf) then
          l = l - nside(1)
        end if

        fx = M_PI * l / nside(1)
        fx = sin(fx) / fx
        if (l == 0) then
          fx = 1
        end if

        k2 = real(n * n + m * m + l * l, kind=mytype)

        ff = 1.0 / (fx * fy * fz)
        gf = pref * exp(-k2 * ismth2) * ff * ff * ff * ff / k2

        if (l == 0 .and. m == 0 .and. n == 0) then
          gf = pref
        end if

        out(i, j, k) = out(i, j, k) * gf
      end do
    end do
  end do

  allocate(in2(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)))

  call decomp_2d_fft_3d(out, in2)

  do k = 0, xsize(3) - 1
    do j = 0, xsize(2) - 1
      do i = 0, xsize(1) - 1
        data((i * xsize(2) + j) * xsize(3) + k + 1) = in2(i + xstart(1), j + xstart(2), k + xstart(3))
      end do
    end do
  end do

  deallocate(in, in2, out)
  call decomp_2d_fft_finalize
  return
end subroutine convolution
