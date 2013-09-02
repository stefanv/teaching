#
# Cookbook Name:: epd-dependencies
# Recipe:: default
#
# Copyright 2012, YOUR_COMPANY_NAME
case node[:platform_family]
when "debian"
  apt_package "remove_python3" do
    package_name "python3"
    action :purge
  end
  package "emacs"
  package "gdb"
  package "gfortran"
  package "git"
  package "kcachegrind"
  package "libatlas-base-dev"
  package "linux-tools"
  package "python-dev"
  package "python-virtualenv"
  package "python2.7-dbg"
  package "valgrind"
  package "vim"

  bash "checkout_sources" do
    environment 'HOME' => '/home/vagrant', 'USER' => 'vagrant'
    user "vagrant"
    group "vagrant"
    code <<-EOF
    set -e
    mkdir -p ~/src
    cd ~/src && rm -rf numpy-git && git clone https://github.com/numpy/numpy numpy-git
    cd ~/src && rm -rf bento-git && git clone https://github.com/cournape/Bento bento-git
    cd ~/src && rm -rf waf-git && git clone https://code.google.com/p/waf waf-git
    cd ~/src && rm -rf flame-graph-git && git clone https://github.com/brendangregg/FlameGraph flame-graph-git
    touch ~/src/checkout_done
    EOF
    not_if { ::File.exists?("/home/vagrant/src/checkout_done") }
  end

  bash "allow_debug" do
    user "root"
    code <<-EOF
    echo 0 > /proc/sys/kernel/kptr_restrict
    EOF
  end
end
