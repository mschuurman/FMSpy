!
!
! Evaluates integrals of arbitary moments of Gaussian basis functions
!
!

  !
  ! Returns the matrix element <cmplx_gaus(q,p)| q^N |cmplx_gaus(q,p)>
  !   -- up to an overlap integral --
  !
  subroutine qn_integral(n, a1, x1, p1, a2, x2, p2, int_qn)
    implicit none
    integer, intent(in)          :: n
    double precision, intent(in) :: a1,x1,p1,a2,x2,p2
    double complex, intent(out)  :: int_qn

    integer          :: i, n_2
    double precision :: a
    double complex   :: b
    double complex, parameter    :: Im = (0., 1.)
    double precision, parameter  :: zero = 1.e-16

    n_2    = int(floor(0.5*n))
    a      = a1 + a2
    b      = 2.*(a1*x1 + a2*x2) - Im*(p1-p2)

    int_qn = (0., 0.)

    if (abs(b) < zero) then
      if (mod(n,2) == 0) then 
        int_qn = a**(-n_2) * factorial(n) / (factorial(n_2) * 2.**n)
      endif
      return
    endif
  
    do i = 0,n_2
      int_qn = int_qn + a**(i-n) * b**(n-2*i) / (factorial(i) * factorial(n-2*i))
    enddo
    int_qn = int_qn * factorial(n) / 2.**n

    return

  contains
    recursive function factorial(n) result (fac)
      integer, intent(in)  :: n
      integer              :: fac

      if (n==0) then
         fac = 1
      else
         fac = n * factorial(n-1)
      endif
    end function factorial

  end subroutine qn_integral

  !
  ! Returns the matrix elements <cmplx_gaus(q1i,p1i)| q^N |cmplx_gaus(q2i,p2i)>
  !   -- up to an overlap integral, as a vector for each (x1,p1),(x2,p2) element
  !
  subroutine qn_vector(n, a1, x1, p1, a2, x2, p2, ints_qn)
    implicit none
    integer, intent(in)          :: n
    double precision, intent(in) :: a1(:),x1(:),p1(:),a2(:),x2(:),p2(:)
    double complex, intent(out)  :: ints_qn(size(a1))

    integer        :: i
    double complex :: int_qn

    ints_qn = (0., 0.)
    do i = 1, size(ints_qn)
       call qn_integral(n,a1(i),x1(i),p1(i),a2(i),x2(i),p2(i),int_qn)
       ints_qn(i) = int_qn
    enddo

    return

  end subroutine qn_vector
